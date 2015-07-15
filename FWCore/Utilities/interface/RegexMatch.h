#ifndef FWCore_Utilities_RegexMatch_h
#define FWCore_Utilities_RegexMatch_h

#include <regex>
#include <string>
#include <vector>

namespace edm {

  bool
  untaintString(char const* pattern, char const* regexp);

  bool
  is_glob(std::string const& pattern);

  std::string
  glob2reg(std::string const& pattern);

  std::vector<std::vector<std::string>::const_iterator>
  regexMatch(std::vector<std::string> const& strings, std::regex const& regexp);

  std::vector<std::vector<std::string>::const_iterator>
  regexMatch(std::vector<std::string> const& strings, std::string const& pattern);
}

#endif // FWCore_Utilities_RegexMatch_h
