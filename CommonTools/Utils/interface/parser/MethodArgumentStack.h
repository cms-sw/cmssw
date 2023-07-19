#ifndef CommonTools_Utils_MethodArgumentStack_h
#define CommonTools_Utils_MethodArgumentStack_h

/* \class reco::parser::MethodArgumentStack
 *
 * Stack of method arguments
 *
 * \author  Giovanni Petrucciani, SNS
 *
 */

#include "CommonTools/Utils/interface/parser/MethodInvoker.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<AnyMethodArgument> MethodArgumentStack;
  }  // namespace parser
}  // namespace reco

#endif  // CommonTools_Utils_MethodArgumentStack_h
