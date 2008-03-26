#ifndef Utilities_FunctionSetter_h
#define Utilities_FunctionSetter_h
/* \class reco::parser::FunctionSetter
 *
 * Function setter
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/Function.h"
#include "PhysicsTools/Utilities/src/FunctionStack.h"

namespace reco {
  namespace parser {    
    struct FunctionSetter {
      FunctionSetter( Function fun, FunctionStack& stack ):
	fun_( fun ), stack_( stack ) {}
      
      void operator()( const char *, const char * ) const { 
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing math function: " << functionNames[ fun_ ] << std::endl;
#endif
	stack_.push_back( fun_ ); 
      }
    private:
      Function fun_;
      FunctionStack & stack_;
    };
  }
}

#endif
