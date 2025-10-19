from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv



load_dotenv(".env")
google_api_key = os.getenv("OPENAI_API_KEY")

gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model_name = "gemini-2.0-flash"



LEADS_FILE = "leads.csv"
FEEDBACK_FILE = "feedback.csv"


def record_customer_interest(name: str, email: str, message: str = "No Message provided") -> str:
    # Logs customer interest to a CSV file.

    # Default unknown fields if missing
    name = name.strip() if name else "Unknown"
    email = email.strip() if email else "Unknown"
    message = message.strip() if message else "No message provided"

    # Prepare new lead row
    new_lead = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "email": email,
        "message": message
    }

    # Append to CSV
    if not os.path.exists(LEADS_FILE):
        df = pd.DataFrame([new_lead])
        df.to_csv(LEADS_FILE, index=False)
    else:
        df = pd.read_csv(LEADS_FILE)
        df = pd.concat([df, pd.DataFrame([new_lead])], ignore_index=True)
        df.to_csv(LEADS_FILE, index=False)

    return f"Thank you {name}! We'll contact you soon."

def record_feedback(question: str) -> str:
    """Logs unknown questions to a CSV file for later review."""

    question = question.strip() if question else "Empty question"

    new_feedback = {
        "timestamp": datetime.now().isoformat(),
        "question": question
    }

    # Append to CSV
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame([new_feedback])
        df.to_csv(FEEDBACK_FILE, index=False)
    else:
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([new_feedback])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)

    return "Thank you! We've logged your question and will review it soon."

record_customer_interest_json = {
    "name": "record_customer_interest",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "message": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_feedback_json = {
    "name": "record_feedback",
    "description": "Use this tool when the bot does not know the answer to a user's question. Logs the question for review.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question the user asked that the bot could not answer"
            },
            "context": {
                "type": "string",
                "description": "Optional context from the conversation that could help later in understanding the question"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function","function": record_customer_interest_json},
        {"type": "function","function": record_feedback_json}]


def handle_tool_calls(tool_calls):
    """
    Execute any tool call returned by the model dynamically.

    Args:
        tool_calls: List of tool call objects returned by Gemini

    Returns:
        List of dictionaries with results from each tool call
    """
    results = []

    for tool_call in tool_calls:
        # Extract tool name and arguments
        tool_name = tool_call.function.name  # e.g., "record_customer_interest"
        arguments = json.loads(tool_call.function.arguments)  # convert JSON string to dict
       

        # Dynamically get the Python function from globals()
        tool_function = globals().get(tool_name)
        if tool_function:
            result = tool_function(**arguments)
            
        else:
            result = {"error": f"No function found for {tool_name}"}

        # Append structured result
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": getattr(tool_call, "id", None)
        })

    return results


reader = PdfReader("me/about_business.pdf")
business = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        business += text

with open("me/summary.txt", "r", encoding="utf-8") as f:  # summary path
    summary = f.read()

name = "Zaynab" 

system_prompt = f"""
You are acting as Bloom & Vows Floral Wedding. You are answering questions on the Bloom & Vows website,
particularly questions related to the business, services, team, mission, and offerings.
Your responsibility is to represent Bloom & Vows faithfully for interactions with potential clients.
You are given a summary of the business and the full business profile which you can use to answer questions.
Be professional, warm, and engaging, as if talking to a potential client planning their wedding.
If you don't know the answer to any question, use your record_feedback tool to record the question that you couldn't answer,
even if it's about something trivial or unrelated to weddings.
If the user is engaging in discussion, try to steer them towards leaving their contact info;
ask for their name and email, and record it using your record_customer_interest tool.
"""

# Append the business summary and full PDF text
system_prompt += f"\n\n## Business Summary:\n{summary}\n\n## Full Business Profile:\n{business}\n\n"
system_prompt += "With this context, please chat with the user, always staying in character as Bloom & Vows Floral Wedding."



def chat(user_input, chat_history):
    # Start with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Append previous chat history as proper dicts
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Add current user input
    messages.append({"role": "user", "content": user_input})

    done = False
    while not done:
        # Call LLM
        response = gemini.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools
        )

        finish_reason = response.choices[0].finish_reason

        # Handle tool calls if needed
        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True

    reply_text = response.choices[0].message.content
    return reply_text


def respond(user_input, chat_history):
    reply = chat(user_input, chat_history)
    chat_history.append((user_input, reply))
    return chat_history, chat_history, ""  

history = []

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌷 Bloom & Vows Personal Assistant
        Welcome to your floral concierge — here to help with wedding bouquets, event florals, and all your dreamy arrangements 💐
        """
    )

    chatbot = gr.Chatbot(label="Bloom & Vows Chat")
    msg = gr.Textbox(placeholder="Type your message here...")
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, chatbot, msg], postprocess=lambda _: "")


if __name__ == "__main__":
    demo.launch()

