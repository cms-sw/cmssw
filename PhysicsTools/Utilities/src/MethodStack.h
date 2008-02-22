#ifndef PhysicsTools_Utilities_MethodStack_h
#define PhysicsTools_Utilities_MethodStack_h
/* \class reco::parser::MethodStack
 *
 * Stack of reflex methods
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<MethodInvoker> MethodStack;
  }
}

#endif
