#ifndef PhysicsTools_Utilities_TypeStack_h
#define PhysicsTools_Utilities_TypeStack_h
/* \class reco::parser::TypeStack
 *
 * Stack of reflex methods
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "Reflex/Type.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<ROOT::Reflex::Type> TypeStack;
  }
}

#endif
