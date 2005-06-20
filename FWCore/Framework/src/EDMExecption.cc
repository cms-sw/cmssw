
#include "FWCore/CoreFramework/interface/EDMException.h"

#define MAP_ENTRY(name) trans_[edm::errors::name]=#name

namespace edm
{
  template<> void Exception::loadTable()
  {
    MAP_ENTRY(NotSuccess);
    MAP_ENTRY(Badness);
    MAP_ENTRY(GreatBadness);
  }
 
}
