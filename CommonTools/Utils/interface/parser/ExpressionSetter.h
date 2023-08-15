#ifndef CommonTools_Utils_ExpressionSetter_h
#define CommonTools_Utils_ExpressionSetter_h
/* \class reco::parser::ExpressionSetter
 *
 * Expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/ExpressionPtr.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"

namespace reco {
  namespace parser {
    struct ExpressionSetter {
      ExpressionSetter(ExpressionPtr& expr, ExpressionStack& exprStack) : expr_(expr), exprStack_(exprStack) {}

      void operator()(const char*, const char*) const;
      ExpressionPtr& expr_;
      ExpressionStack& exprStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif
