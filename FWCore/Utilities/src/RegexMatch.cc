// functions used to assist with  regular expression matching of strings

#include "FWCore/Utilities/interface/RegexMatch.h"

#include <boost/algorithm/string.hpp>
#include <regex>

namespace edm {

  // checks that a tainted string is valid.
  // Needed to satisfy Coverity.
  bool
  untaintString(char const* pattern, char const* regexp) {
    std::regex rexp(regexp);
    return std::regex_match(pattern, rexp);
  }

  bool is_glob(std::string const& pattern) {
    return (pattern.find_first_of("*?") != pattern.npos);
  }

  std::string glob2reg(std::string const& pattern) {
    std::string regexp = pattern;
    boost::replace_all(regexp, "*", ".*");
    boost::replace_all(regexp, "?", ".");
    return regexp;
  }

  std::vector<std::vector<std::string>::const_iterator>
  regexMatch(std::vector<std::string> const& strings, std::regex const& regexp) {
    std::vector< std::vector<std::string>::const_iterator> matches;
    for (std::vector<std::string>::const_iterator i = strings.begin(), iEnd = strings.end(); i != iEnd; ++i) {
      if (std::regex_match((*i), regexp)) {
        matches.push_back(i);
      }
    }
    return matches;
  }

  std::vector<std::vector<std::string>::const_iterator>
  regexMatch(std::vector<std::string> const& strings, std::string const& pattern) {
    std::regex regexp(glob2reg(pattern));
    return regexMatch(strings, regexp);
  }

}
