// functions used to assist with  regular expression matching of strings

#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

#include "FWCore/Utilities/interface/RegexMatch.h"

namespace edm {
  
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
  regexMatch(std::vector<std::string> const& strings, boost::regex const& regexp) {
    std::vector< std::vector<std::string>::const_iterator> matches;
    for (std::vector<std::string>::const_iterator i = strings.begin(), iEnd = strings.end(); i != iEnd; ++i) {
      if (boost::regex_match((*i), regexp)) {
        matches.push_back(i);
      }
    }
    return matches;
  }
  
  std::vector<std::vector<std::string>::const_iterator> 
  regexMatch(std::vector<std::string> const& strings, std::string const& pattern) {
    boost::regex regexp(glob2reg(pattern));
    return regexMatch(strings, regexp);
  }
  
}
