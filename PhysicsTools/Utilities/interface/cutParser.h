#ifndef PhysicsTools_Utilities_cutParset_h
#define PhysicsTools_Utilities_cutParset_h
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "PhysicsTools/Utilities/src/SelectorPtr.h"
#include <string>

namespace reco {
  namespace parser {
    bool cutParser( const std::string &, const MethodMap &, SelectorPtr & );
  }
}

#endif
