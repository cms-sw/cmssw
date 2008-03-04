#ifndef Utilities_ExpressionNumberSetter_h
#define Utilities_ExpressionNumberSetter_h
/* \class reco::parser::ExpressionNumber
 *
 * Numerical expression setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/ExpressionNumber.h"
#include "PhysicsTools/Utilities/src/ExpressionStack.h"
#ifdef BOOST_SPIRIT_DEBUG 
#include <iostream>
#endif
namespace reco {
  namespace parser {
    struct ExpressionNumberSetter {
      ExpressionNumberSetter( ExpressionStack & stack ) : stack_( stack ) { }
      void operator()( double n ) const {
#ifdef BOOST_SPIRIT_DEBUG 
  BOOST_SPIRIT_DEBUG_OUT << "pushing number: " << n << std::endl;
#endif
	stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionNumber( n ) ) );
      }
    private:
      ExpressionStack & stack_;
    };
  }
}

#endif
