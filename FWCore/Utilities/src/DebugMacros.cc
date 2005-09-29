
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <cstdlib>

using namespace std;

namespace edm {

  debugvalue::debugvalue():
    cvalue_(getenv("PROC_DEBUG")),
    value_(cvalue_==0 ? 0 : atoi(cvalue_))
  { }
  
  debugvalue debugit;
  
}
