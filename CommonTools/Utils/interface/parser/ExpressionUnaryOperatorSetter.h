#ifndef CommonTools_Utils_ExpressionUnaryOperatorSetter_h
#define CommonTools_Utils_ExpressionUnaryOperatorSetter_h
/* \class reco::parser::ExpressionUnaryOperator
 *
 * Unary Operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/ExpressionUnaryOperator.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"
#ifdef BOOST_SPIRIT_DEBUG
#include <string>
#include <iostream>
#endif
namespace reco {
  namespace parser {

#ifdef BOOST_SPIRIT_DEBUG
    template <typename Op>
    struct op1_out {
      static const std::string value;
    };
#endif

    template <typename Op>
    struct ExpressionUnaryOperatorSetter {
      ExpressionUnaryOperatorSetter(ExpressionStack& stack) : stack_(stack) {}
      void operator()(const char*, const char*) const {
#ifdef BOOST_SPIRIT_DEBUG
        BOOST_SPIRIT_DEBUG_OUT << "pushing unary operator" << op1_out<Op>::value << std::endl;
#endif
        stack_.push_back(ExpressionPtr(new ExpressionUnaryOperator<Op>(stack_)));
      }

    private:
      ExpressionStack& stack_;
    };
  }  // namespace parser
}  // namespace reco

#endif
