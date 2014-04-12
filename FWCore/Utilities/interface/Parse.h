#ifndef FWCore_Utilities_Parse_h
#define FWCore_Utilities_Parse_h

#include <string>
#include <vector>
namespace edm {
      /// only does the yacc interpretation
      std::string read_whole_file(std::string const& filename);

      void read_from_cin(std::string & output);

      std::string withoutQuotes(std::string const& from);

      /// breaks the input string into tokens, delimited by the separator
      std::vector<std::string> tokenize(std::string const& input, std::string const& separator);
}

#endif
