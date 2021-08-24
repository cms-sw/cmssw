#ifndef CommonTools_Utils_TypeStack_h
#define CommonTools_Utils_TypeStack_h
/* \class reco::parser::TypeStack
 *
 * Stack of types
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<edm::TypeWithDict> TypeStack;
  }
}  // namespace reco

#endif
