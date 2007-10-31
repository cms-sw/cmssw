#ifndef Utilities_NotCombiner_h
#define Utilities_NotCombiner_h
/* \class reco::parser::NotCombiner
 *
 * logical NOT combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "PhysicsTools/Utilities/src/SelectorBase.h"
#include "PhysicsTools/Utilities/src/SelectorPtr.h"

namespace reco {
  namespace parser {    
    struct NotCombiner : public SelectorBase {
      NotCombiner( SelectorPtr arg ) :
	arg_( arg ) {}
      virtual bool operator()( const ROOT::Reflex::Object& o ) const {
	return ! (*arg_)( o );
      }
    private:
      SelectorPtr arg_;
    };
  }
}

#endif
