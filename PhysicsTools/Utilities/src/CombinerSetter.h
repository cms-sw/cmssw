#ifndef Utilities_CombinerSetter_h
#define Utilities_CombinerSetter_h
/* \class reco::parser::CombinerSetter
 *
 * Combiner setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/Combiner.h"
#include "PhysicsTools/Utilities/src/CombinerStack.h"

namespace reco {
  namespace parser {    
    struct CombinerSetter {
      CombinerSetter(Combiner comb, CombinerStack& stack):
	comb_(comb), stack_(stack) {}
      
      void operator()(const char * const &, const char * const &) const { 
	stack_.push_back(comb_); 
      }
      void operator()(const char &) const {
	stack_.push_back(comb_); 
      }
    private:
      Combiner comb_;
      CombinerStack & stack_;
    };
  }
}

#endif
