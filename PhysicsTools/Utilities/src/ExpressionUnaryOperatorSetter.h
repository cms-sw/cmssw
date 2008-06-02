#ifndef Utilities_ExpressionUnaryOperatorSetter_h
#define Utilities_ExpressionUnaryOperatorSetter_h
/* \class reco::parser::ExpressionUnaryOperator
 *
 * Unary Operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionUnaryOperator.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#ifdef BOOST_SPIRIT_DEBUG 
#include <string>
#endif
namespace reco {
  namespace parser {

#ifdef BOOST_SPIRIT_DEBUG 
    template <typename Op> struct op1_out { static const std::string value; };
#endif

    template<typename Op>
    struct ExpressionUnaryOperatorSetter {
      ExpressionUnaryOperatorSetter( ExpressionStack & stack ) : stack_( stack ) { }
      void operator()( const char*, const char* ) const {
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing unary operator" << op1_out<Op>::value << std::endl;
#endif	
	stack_.push_back( ExpressionPtr( new ExpressionUnaryOperator<Op>( stack_ ) ) );
      }
    private:
      ExpressionStack & stack_;
    };
  }
}

#endif
