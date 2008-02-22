#ifndef Utilities_ExpressionBinaryOperator_h
#define Utilities_ExpressionBinaryOperator_h
/* \class reco::parser::ExpressionBinaryOperator
 *
 * Binary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionBase.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"

namespace reco {
  namespace parser {
    template<typename Op>
    struct ExpressionBinaryOperator : public ExpressionBase {
      virtual double value( const ROOT::Reflex::Object& o ) const { 
	return op_( (*lhs_).value( o ), (*rhs_).value( o ) );
      }
      ExpressionBinaryOperator( ExpressionStack & expStack ) { 
	rhs_ = expStack.back(); expStack.pop_back();
	lhs_ = expStack.back(); expStack.pop_back();
      }
    private:
      Op op_;
      ExpressionPtr lhs_, rhs_;
    };
  }
}

#endif
