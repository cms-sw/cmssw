#ifndef Utilities_ExpressionBinaryOperatorSetter_h
#define Utilities_ExpressionBinaryOperatorSetter_h
/* \class reco::parser::ExpressionBinaryOperator
 *
 * Binary operator expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionBinaryOperator.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#include <cmath>

namespace reco {
  namespace parser {

#ifdef BOOST_SPIRIT_DEBUG 
    template <typename Op> struct op2_out { static const char value; };
#endif

    template<typename T>
    struct power_of {
      T operator()( T lhs, T rhs ) const { return pow( lhs, rhs ); }
    };

    template<typename Op>
    struct ExpressionBinaryOperatorSetter {
      ExpressionBinaryOperatorSetter( ExpressionStack & stack ) : stack_( stack ) { }
      void operator()( const char*, const char* ) const {
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing binary operator: " << op2_out<Op>::value << std::endl;
#endif
	stack_.push_back( ExpressionPtr( new ExpressionBinaryOperator<Op>( stack_ ) ) );
      }
    private:
      ExpressionStack & stack_;
    };
  }
}

#endif
