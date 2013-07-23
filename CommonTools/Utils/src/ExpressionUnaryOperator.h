#ifndef CommonTools_Utils_ExpressionUnaryOperator_h
#define CommonTools_Utils_ExpressionUnaryOperator_h
/* \class reco::parser::ExpressionUnaryOperator
 *
 * Unary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/ExpressionStack.h"

namespace reco {
  namespace parser {
    template<typename Op>
    struct ExpressionUnaryOperator : public ExpressionBase {
      virtual double value(const edm::ObjectWithDict& o) const { 
	return op_((*exp_).value(o));
      }
      ExpressionUnaryOperator(ExpressionStack & expStack) { 
	exp_ = expStack.back(); expStack.pop_back();
      }
    private:
      Op op_;
      ExpressionPtr exp_;
    };
  }
}

#endif
