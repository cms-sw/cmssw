#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/src/ConfigurationPreprocessor.h"
#include "FWCore/ParameterSet/src/ParseResultsTweaker.h"
#include <fstream>
#include <iostream>

// the parse() method comes from the yacc file, not here

namespace edm {
  namespace pset {

    ParseResults fullParse(const std::string & input) 
    {
      // preprocess, for things like 'include'
      std::string preprocessedConfigString;
      ConfigurationPreprocessor preprocessor;
      preprocessor.process(input, preprocessedConfigString);

      boost::shared_ptr<edm::pset::NodePtrList> parsetree =
      edm::pset::parse(preprocessedConfigString.c_str());

       // postprocess, for things like replace and rename
      ParseResultsTweaker tweaker;
      tweaker.process(parsetree);
    
      return parsetree;
    }


    bool read_whole_file(std::string const& filename, std::string& output)
    {
      std::ifstream input(filename.c_str());
      if (!input) return false;
      std::string buffer;
      while (std::getline(input, buffer))
        {
          // getline strips newlines; we have to put them back by hand.
          output += buffer;
          output += '\n';
        }
      return true;
    }


    void read_from_cin(std::string & output) 
    {
      std::string line;
      while (std::getline(std::cin, line))
      {
        output += line;
        output += '\n';
      }
    }

  }  // namespace pset
} // namespace edm

