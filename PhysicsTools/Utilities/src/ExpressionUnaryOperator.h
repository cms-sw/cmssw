#ifndef Utilities_ExpressionUnaryOperator_h
#define Utilities_ExpressionUnaryOperator_h
/* \class reco::parser::ExpressionUnaryOperator
 *
 * Unary Operator expression
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
    struct ExpressionUnaryOperator : public ExpressionBase {
      virtual double value( const ROOT::Reflex::Object& o ) const { 
	return op_( ( *exp_).value( o ) );
      }
      ExpressionUnaryOperator( ExpressionStack & expStack ) { 
	exp_ = expStack.back(); expStack.pop_back();
      }
    private:
      Op op_;
      ExpressionPtr exp_;
    };
  }
}

#endif
