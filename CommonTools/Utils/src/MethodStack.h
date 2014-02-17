#ifndef CommonTools_Utils_MethodStack_h
#define CommonTools_Utils_MethodStack_h
/* \class reco::parser::MethodStack
 *
 * Stack of methods
 *
 * \author  Luca Lista, INFN
 *
 * \version $Revision: 1.4 $
 *
 */
#include "CommonTools/Utils/src/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<MethodInvoker> MethodStack;
    typedef std::vector<LazyInvoker>   LazyMethodStack;
  }
}

#endif
