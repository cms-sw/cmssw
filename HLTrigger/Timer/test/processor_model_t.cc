#include <cctype>
#include <iostream>

#include "HLTrigger/Timer/interface/processor_model.h"

const char* article(char letter) {
  switch (tolower(letter)) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
      return "an";
    default:
      return "a";
  }
}

const char* article(const char* word) { return word == nullptr ? nullptr : article(word[0]); }

const char* article(const std::string& word) { return word.empty() ? nullptr : article(word[0]); }

int main() {
  std::cout << "Running on " << article(processor_model) << " " << processor_model << std::endl;

  return processor_model.empty();
}
