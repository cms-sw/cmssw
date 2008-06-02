#ifndef Utilities_CombinerSetter_h
#define Utilities_CombinerSetter_h
/* \class reco::parser::CombinerSetter
 *
 * Combiner setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/Combiner.h"
#include "PhysicsTools/Utilities/src/CombinerStack.h"

namespace reco {
  namespace parser {    
    struct CombinerSetter {
      CombinerSetter( Combiner comb, CombinerStack& stack ):
	comb_( comb ), stack_( stack ) {}
      
      void operator()( const char & ) const { 
#ifdef BOOST_SPIRIT_DEBUG 
	BOOST_SPIRIT_DEBUG_OUT << "pushing logical combiner: ";
	if ( comb_ == kAnd )
	  BOOST_SPIRIT_DEBUG_OUT << '&';
	else if( comb_ == kOr )
	  BOOST_SPIRIT_DEBUG_OUT << '|';
	else if( comb_ == kNot )
	  BOOST_SPIRIT_DEBUG_OUT << '!';
	BOOST_SPIRIT_DEBUG_OUT << std::endl;
#endif
	stack_.push_back( comb_ ); 
      }
    private:
      Combiner comb_;
      CombinerStack & stack_;
    };
  }
}

#endif
