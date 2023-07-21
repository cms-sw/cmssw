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
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"

namespace reco {
  namespace parser {
    template <typename Op>
    struct ExpressionUnaryOperator : public ExpressionBase {
      double value(const edm::ObjectWithDict& o) const override { return op_((*exp_).value(o)); }
      ExpressionUnaryOperator(ExpressionStack& expStack) {
        exp_ = expStack.back();
        expStack.pop_back();
      }

    private:
      Op op_;
      ExpressionPtr exp_;
    };
  }  // namespace parser
}  // namespace reco

#endif
