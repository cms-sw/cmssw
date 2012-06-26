#ifndef CommonTools_Utils_TypeStack_h
#define CommonTools_Utils_TypeStack_h
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
    typedef std::vector<Reflex::Type> TypeStack;
  }
}

#endif
