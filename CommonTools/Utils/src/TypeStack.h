#ifndef CommonTools_Utils_TypeStack_h
#define CommonTools_Utils_TypeStack_h
/* \class reco::parser::TypeStack
 *
 * Stack of types
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<edm::TypeWithDict> TypeStack;
  }
}

#endif
