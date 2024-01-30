# Yuval Saadaty 205956634
import os
import sys
from collections import Counter
from collections import Counter
import math

if len(sys.argv) < 4:
    print("Misssing argumnets <development set filename> <test set filename> <INPUT_WORD> <output filename>")
    sys.exit(1)

development_set_filename = os.path.basename(sys.argv[1]) # first argument - develop set file name
test_set_filename = os.path.basename(sys.argv[2]) # second argument - test set file name
input_word = os.path.basename(sys.argv[3]) # third argument - a simple string representing either a seen or unseen word
output_filename = os.path.basename(sys.argv[4]) # fourth argument - output file name
output = [] # list of output lines to be written to the output file
output += ["#Student Yuval Saadaty 205956634\n"] # Student name
output += [f"#Output1 {development_set_filename} \n" ] # development set file name
output += [f"#Output2 {test_set_filename}\n"] # test set file name
output += [f"#Output3 {input_word} \n"] # input word
output += [f"#Output4 {output_filename}\n"] # output file name

vocabulary_size = 300000 # vocabulary size given in the assignment description
output += [f"#Output5 {vocabulary_size}\n"] # vocabulary size
output += [f"#Output6 {(1/vocabulary_size)}\n"] # the probability of this event in the uniform distribution

development_set_events = [] # list of events from the development set 
with open(development_set_filename) as file:
    for (index, line) in enumerate(file):
        if index == 2 or (index - 2) % 4 == 0:
            development_set_events += line.split()

# number of events in the development set
development_set_events_size = len(development_set_events) 
output += [f"#Output7 {development_set_events_size}\n"] # total number of events in the development set |S|

# Select the first 10% of words in the development set as the validation set
validation_set_text = []
validation_set_events_size = round(0.1 * development_set_events_size)
output += [f"#Output8 {validation_set_events_size}\n"]

# Select the first 90% of words in the development set as the training set
training_set_text = []
training_set_events_size = development_set_events_size-validation_set_events_size
training_set_text.extend(development_set_events[:training_set_events_size])
output += [f"#Output9 {training_set_events_size}\n"]
unique_events_in_training_set = Counter(training_set_text) # count the occurrences of each word in the training set
unique_events_in_training_set_size = len(Counter(training_set_text)) # count the occurrences of each word in the training set
output += [f"#Output10 {unique_events_in_training_set_size}\n"]

# number of occurrences of input_word in the training set
input_word_in_training_set = training_set_text.count(input_word) 
output += [f"#Output11 {input_word_in_training_set}\n"]


# Calculate MLE for input_word
input_word_trainig_mle_no_smoothing = input_word_in_training_set/len(training_set_text)
output += [f"#Output12 {input_word_trainig_mle_no_smoothing}\n"]

# Calculate MLE for unseen word
unseenword_in_training_set = "kids"
unseenword_in_training_set = training_set_text.count(unseenword_in_training_set)/len(training_set_text)
output += [f"#Output13 {unseenword_in_training_set}\n"]


# Train the Lidstone unigram model with smoothing parameter (λ)
def lidstone_probability(word, smoothing_parameter):
    return (unique_events_in_training_set[word] + smoothing_parameter) / (training_set_events_size + (smoothing_parameter * vocabulary_size))

# Train the Lidstone unigram model
input_word_in_lidstone_model = lidstone_probability(input_word, 0.1)
# P(Event = INPUT WORD) as estimated by your model using λ = 0.1
output += [f"#Output14 {input_word_in_lidstone_model}\n"]

# Train the Lidstone unigram model
unseenword_in_lidstone_model = lidstone_probability(unseenword_in_training_set, 0.1)
# P(Event = INPUT WORD) as estimated by your model using λ = 0.1
output += [f"#Output15 {unseenword_in_lidstone_model}\n"]

validation_set_text.extend(development_set_events[training_set_events_size:development_set_events_size])
def calculate_perplexity(smoothing_parameter, data_set):
    total_log_probability = sum([math.log(lidstone_probability(word, smoothing_parameter), 2) for word in data_set])
    average_log_probability = total_log_probability / len(data_set)
    perplexity = 2 ** (-average_log_probability)
    return perplexity

# Calculate perplexity on the validation set with smoothing parameter λ = 0.01
output += [f"#Output16 {calculate_perplexity(0.01, validation_set_text)}\n"]

# Calculate perplexity on the validation set with smoothing parameter λ = 0.1
output += [f"#Output17 {calculate_perplexity(0.1, validation_set_text)}\n"]

# Calculate perplexity on the validation set with smoothing parameter λ = 1.0
output += [f"#Output18 {calculate_perplexity(1, validation_set_text)}\n"]

def find_optimal_smoothing_parameter( min_lambda, max_lambda):
    best_lambda = None
    min_perplexity = float('inf')
    min_lambda = 1 if min_lambda <= 0 else min_lambda
    for smoothing_parameter in range(min_lambda, int(max_lambda * 100), 1):
        smoothing_parameter /= 100
        perplexity = calculate_perplexity(smoothing_parameter, validation_set_text)

        if perplexity < min_perplexity:
            min_perplexity = perplexity
            best_lambda = smoothing_parameter

    return best_lambda, min_perplexity

# Set the range of lambda values to check
min_lambda = 0
max_lambda = 2
step = 0.01

# Find the optimal smoothing parameter
optimal_lambda, min_perplexity = find_optimal_smoothing_parameter(min_lambda, max_lambda)

#  The value of λ that you found to minimize the perplexity on the validation set
output += [f"#Output19 {optimal_lambda}\n"]

# The minimized perplexity on the validation set using the best value you found for λ
output += [f"#Output20 {min_perplexity}\n"]

test_set_text = []
with open(test_set_filename) as file:
    for index, line in enumerate(file):
        if index == 2 or (index - 2) % 4 == 0:
            test_set_text += line.split()

#  Total number of events in the test set
test_set_events_size = len(test_set_text)
output += [f"#Output21 {test_set_events_size}\n"]

#  The perplexity of the test set according to the Lidstone model with the best λ 
test_set_events_size = len(test_set_text)
output += [f"#Output22 {calculate_perplexity(optimal_lambda, test_set_text)}\n"]

with open (output_filename, 'w') as file:
    file.writelines(output)

