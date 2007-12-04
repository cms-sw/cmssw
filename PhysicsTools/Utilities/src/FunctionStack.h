#ifndef Utilities_FunctionStack_h
#define Utilities_FunctionStack_h
/* \class reco::parser::FunctionStack
 *
 * Function stack
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/Function.h"
#include <vector>

namespace reco {
  namespace parser {    
    typedef std::vector<Function> FunctionStack;
  }
}

#endif
