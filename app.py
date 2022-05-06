#imports
from flask import Flask, render_template, request
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

#create chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response   
def get_bot_response():
    human_text = request.args.get('msg')
    human_text = human_text.lower()
    if human_text != 'bye' or human_text != 'goodbye' or human_text != 'good bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            return ">>A.I. Bot: Most welcome"
        elif human_text == '' or human_text == ' ':
            return ">>A.I. Bot: Please enter something."
        else:
            new_user_input_ids = tokenizer.encode(human_text + tokenizer.eos_token, return_tensors='pt')

            chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            
            return ">>A.I. Bot: " + str(tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True))
        
    else:
        return ">>A.I. Bot: Good bye and take care of yourself..."

if __name__ == "__main__":
    app.run()