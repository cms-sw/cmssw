#ifndef CommonTools_Utils_ExpressionFunctionSetter_h
#define CommonTools_Utils_ExpressionFunctionSetter_h
/* \class reco::parser::ExpressionFunction
 *
 * Numerical expression setter
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "CommonTools/Utils/interface/ExpressionStack.h"
#include "CommonTools/Utils/interface/FunctionStack.h"

namespace reco {
  namespace parser {
    struct ExpressionFunctionSetter {
      ExpressionFunctionSetter(ExpressionStack &expStack, FunctionStack &funStack)
          : expStack_(expStack), funStack_(funStack) {}
      void operator()(const char *, const char *) const;

    private:
      ExpressionStack &expStack_;
      FunctionStack &funStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif
