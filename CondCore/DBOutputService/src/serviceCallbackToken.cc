#include "serviceCallbackToken.h"
#include <boost/functional/hash/hash.hpp>
size_t cond::service::serviceCallbackToken::build(const std::string& str){
  boost::hash< std::string > string_hash;
  size_t tok=string_hash(str);
  return tok;
}
