#ifndef ParameterSet_parse_h
#define ParameterSet_parse_h

#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/Node.h"
typedef boost::shared_ptr<edm::pset::NodePtrList> ParseResults;

namespace edm {
   namespace pset {
      /// does preprocessing, yacc parsing, and postprocessing
      ParseResults fullParse(const std::string & input);

      /// only does the yacc interpretation
      ParseResults parse(char const* spec);

      bool read_whole_file(std::string const& filename, std::string& output);

      void read_from_cin(std::string & output);

      std::string withoutQuotes(const std::string& from);

      /// breaks the input string into tokens, delimited by the separator
      std::vector<std::string> tokenize(const std::string & input, const std::string & separator);
   }
}

#endif

