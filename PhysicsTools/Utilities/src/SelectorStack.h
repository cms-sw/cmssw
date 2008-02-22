#ifndef Utilities_SelectorStack_h
#define Utilities_SelectorStack_h
/* \class reco::parser::SelectorPtr
 *
 * Stack of selectors
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include <vector>
#include "PhysicsTools/Utilities/src/SelectorPtr.h"

namespace reco {
  namespace parser {
    typedef std::vector<SelectorPtr> SelectorStack;
  }
}

#endif
