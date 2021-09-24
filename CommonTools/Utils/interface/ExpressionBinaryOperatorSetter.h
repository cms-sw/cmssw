#ifndef CommonTools_Utils_ExpressionBinaryOperatorSetter_h
#define CommonTools_Utils_ExpressionBinaryOperatorSetter_h
/* \class reco::parser::ExpressionBinaryOperator
 *
 * Binary operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/ExpressionBinaryOperator.h"
#include "CommonTools/Utils/interface/ExpressionStack.h"
#include <cmath>

namespace reco {
  namespace parser {

    template <typename T>
    struct power_of {
      T operator()(T lhs, T rhs) const { return pow(lhs, rhs); }
    };

    template <typename Op>
    struct ExpressionBinaryOperatorSetter {
      ExpressionBinaryOperatorSetter(ExpressionStack& stack) : stack_(stack) {}
      void operator()(const char*, const char*) const {
        stack_.push_back(ExpressionPtr(new ExpressionBinaryOperator<Op>(stack_)));
      }

    private:
      ExpressionStack& stack_;
    };
  }  // namespace parser
}  // namespace reco

#endif
