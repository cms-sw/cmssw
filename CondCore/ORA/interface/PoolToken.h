#ifndef INCLUDE_COND_POOLTOKEN_H
#define INCLUDE_COND_POOLTOKEN_H

#include <string>
#include <utility>

namespace cond {
  
  std::pair<std::string,int> parseToken( const std::string& objectId );
  std::string writeToken( const std::string& containerName, int oid0, int oid1, const std::string& className );
  std::string writeTokenContainerFragment( const std::string& containerName, const std::string& className );
  
}

#endif
