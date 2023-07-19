#ifndef CommonTools_Utils_MethodStack_h
#define CommonTools_Utils_MethodStack_h

/* \class reco::parser::MethodStack
 *
 * Stack of methods
 *
 * \author  Luca Lista, INFN
 *
 */

#include "CommonTools/Utils/interface/parser/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<MethodInvoker> MethodStack;
    typedef std::vector<LazyInvoker> LazyMethodStack;
  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_MethodStack_h
