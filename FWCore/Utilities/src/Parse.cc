#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>

namespace edm {

    std::string  read_whole_file(std::string const& filename) {
      std::string result;
      std::ifstream input(filename.c_str());
      if (!input) {
       throw edm::Exception(errors::Configuration,"MissingFile")
         << "Cannot read file " << filename;
      }
      std::string buffer;
      while (getline(input, buffer)) {
          // getline strips newlines; we have to put them back by hand.
          result += buffer;
          result += '\n';
      }
      return result; 
    }


    void read_from_cin(std::string & output) {
      std::string line;
      while (getline(std::cin, line)) {
        output += line;
        output += '\n';
      }
    }


    std::string withoutQuotes(const std::string& from) {
      std::string result = from;
      if(!result.empty()) {
      // get rid of leading quotes
        if(result[0] == '"' || result[0] == '\'') {
          result.erase(0,1);
        }
      }

      if(!result.empty()) {
       // and trailing quotes
        int lastpos = result.size()-1;
        if(result[lastpos] == '"' || result[lastpos] == '\'') {
          result.erase(lastpos, 1);
        }
      }
     return result;
    }


    std::vector<std::string> 
    tokenize(const std::string & input, const std::string &separator) {
      typedef boost::char_separator<char>   separator_t;
      typedef boost::tokenizer<separator_t> tokenizer_t;

      std::vector<std::string> result;
      separator_t  sep(separator.c_str(), "", boost::keep_empty_tokens); // separator for elements in path
      tokenizer_t  tokens(input, sep);
      copy_all(tokens, std::back_inserter<std::vector<std::string> >(result));
      return result;
    }

} // namespace edm

