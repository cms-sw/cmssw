#include "CondCore/DBOutputService/interface/serviceCallbackToken.h"
#include <utility>
#include <boost/functional/hash/pair.hpp>
size_t cond::service::serviceCallbackToken::build(const std::string& tag,
						 const std::string& container){
  boost::hash< std::pair<std::string, std::string> > pair_hash;
  return pair_hash(std::make_pair(container,tag));
}
