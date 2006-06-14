#include "serviceCallbackToken.h"
#include <utility>
#include <boost/functional/hash/pair.hpp>
size_t cond::service::serviceCallbackToken::build(const std::string& str1,
						  const std::string& str2){
  boost::hash< std::pair<std::string, std::string> > pair_hash;
  return pair_hash(std::make_pair(str1,str2));
}
