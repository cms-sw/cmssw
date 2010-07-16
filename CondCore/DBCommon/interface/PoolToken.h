#ifndef COND_DBCommon_PoolToken_h
#define COND_DBCommon_PoolToken_h

#include <string>
#include <utility>

namespace cond {
  
  std::pair<std::string,int> parseToken( const std::string& objectId );
  std::string writeToken( const std::string& containerName, int oid0, int oid1, const std::string& className );
  
}

#endif
