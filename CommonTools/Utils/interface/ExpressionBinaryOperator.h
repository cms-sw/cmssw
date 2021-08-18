#ifndef CommonTools_Utils_ExpressionBinaryOperator_h
#define CommonTools_Utils_ExpressionBinaryOperator_h
/* \class reco::parser::ExpressionBinaryOperator
 *
 * Binary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/ExpressionBase.h"
#include "CommonTools/Utils/interface/ExpressionStack.h"

namespace reco {
  namespace parser {
    template <typename Op>
    struct ExpressionBinaryOperator : public ExpressionBase {
      double value(const edm::ObjectWithDict& o) const override { return op_((*lhs_).value(o), (*rhs_).value(o)); }
      ExpressionBinaryOperator(ExpressionStack& expStack) {
        rhs_ = expStack.back();
        expStack.pop_back();
        lhs_ = expStack.back();
        expStack.pop_back();
      }

    private:
      Op op_;
      ExpressionPtr lhs_, rhs_;
    };
  }  // namespace parser
}  // namespace reco

#endif
