// functions shared between the edm::EventSelector and the HLTHighLevel filter

#ifndef FWCore_Utilities_RegexMatch_h
#define FWCore_Utilities_RegexMatch_h

#include <vector>
#include <string>

#include <boost/regex_fwd.hpp>

namespace edm {

  bool 
  is_glob(std::string const& pattern);

  std::string
  glob2reg(std::string const& pattern);

  std::vector<std::vector<std::string>::const_iterator> 
  regexMatch(std::vector<std::string> const& strings, boost::regex const& regexp);

  std::vector<std::vector<std::string>::const_iterator> 
  regexMatch(std::vector<std::string> const& strings, std::string const& pattern);

}

#endif // FWCore_Utilities_RegexMatch_h
