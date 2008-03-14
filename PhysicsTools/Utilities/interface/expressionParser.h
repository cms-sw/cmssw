#ifndef PhysicsTools_Utilities_expressionParset_h
#define PhysicsTools_Utilities_expressionParset_h
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include <string>

namespace reco {
  namespace parser {
    bool expressionParser( const std::string &, const MethodMap &, ExpressionPtr & );
  }
}

#endif
