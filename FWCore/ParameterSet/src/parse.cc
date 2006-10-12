#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/src/ConfigurationPreprocessor.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>

// the parse() method comes from the yacc file, not here
using namespace std;

namespace edm {
  namespace pset {

    string  read_whole_file(string const& filename)
    {
      string result;
      ifstream input(filename.c_str());
      if (!input) {
       throw edm::Exception(errors::Configuration,"MissingFile")
         << "Cannot read file " << filename;
      }
      string buffer;
      while (getline(input, buffer))
        {
          // getline strips newlines; we have to put them back by hand.
          result += buffer;
          result += '\n';
        }
      return result; 
    }


    void read_from_cin(string & output) 
    {
      string line;
      while (getline(cin, line))
      {
        output += line;
        output += '\n';
      }
    }


    string withoutQuotes(const string& from)
    {
      string result = from;
      if(!result.empty())
      {
      // get rid of leading quotes
        if(result[0] == '"' || result[0] == '\'')
        {
          result.erase(0,1);
        }
      }

      if(!result.empty())
      {
       // and trailing quotes
        int lastpos = result.size()-1;
        if(result[lastpos] == '"' || result[lastpos] == '\'')
        {
          result.erase(lastpos, 1);
        }
      }
     return result;
    }


    std::vector<std::string> 
    tokenize(const std::string & input, const std::string &separator) 
    {
      typedef boost::char_separator<char>   separator_t;
      typedef boost::tokenizer<separator_t> tokenizer_t;

      std::vector<string> result;
      separator_t  sep(separator.c_str(), "", boost::keep_empty_tokens); // separator for elements in path
      tokenizer_t  tokens(input, sep);
      std::copy(tokens.begin(),
                tokens.end(),
                std::back_inserter<vector<string> >(result));
      return result;
    }

  }  // namespace pset
} // namespace edm

