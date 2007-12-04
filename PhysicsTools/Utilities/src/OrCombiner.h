#ifndef Utilities_OrCombiner_h
#define Utilities_OrCombiner_h
/* \class reco::parser::OrCombiner
 *
 * logical OR combiner
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
    struct OrCombiner : public SelectorBase {
      OrCombiner( SelectorPtr lhs, SelectorPtr rhs ) :
	lhs_( lhs ), rhs_( rhs ) {}
      virtual bool operator()( const ROOT::Reflex::Object& o ) const {
	return (*lhs_)( o ) || (*rhs_)( o );
      }
    private:
      SelectorPtr lhs_, rhs_;
    };
  }
}

#endif
