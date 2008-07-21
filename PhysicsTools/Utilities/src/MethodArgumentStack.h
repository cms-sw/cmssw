#ifndef PhysicsTools_Utilities_MethodArgumentStack_h
#define PhysicsTools_Utilities_MethodArgumentStack_h
/* \class reco::parser::MethodArgumentStack
 *
 * Stack of method arguments
 *
 * \author  Giovanni Petrucciani, SNS
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<AnyMethodArgument> MethodArgumentStack;
  }
}

#endif
