#ifndef CommonTools_Utils_FunctionStack_h
#define CommonTools_Utils_FunctionStack_h
/* \class reco::parser::FunctionStack
 *
 * Function stack
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "CommonTools/Utils/interface/Function.h"
#include <vector>

namespace reco {
  namespace parser {
    typedef std::vector<Function> FunctionStack;
  }
}  // namespace reco

#endif
