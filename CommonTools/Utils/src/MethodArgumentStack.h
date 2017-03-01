#ifndef CommonTools_Utils_MethodArgumentStack_h
#define CommonTools_Utils_MethodArgumentStack_h

/* \class reco::parser::MethodArgumentStack
 *
 * Stack of method arguments
 *
 * \author  Giovanni Petrucciani, SNS
 *
 */

#include "CommonTools/Utils/src/MethodInvoker.h"
#include <vector>

namespace reco {
namespace parser {
typedef std::vector<AnyMethodArgument> MethodArgumentStack;
} // namespace reco
} // namespace parser

#endif // CommonTools_Utils_MethodArgumentStack_h
