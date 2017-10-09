#ifndef CommonTools_Utils_NotCombiner_h
#define CommonTools_Utils_NotCombiner_h
/* \class reco::parser::NotCombiner
 *
 * logical NOT combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/SelectorPtr.h"

namespace reco {
  namespace parser {    
    struct NotCombiner : public SelectorBase {
      NotCombiner( SelectorPtr arg ) :
	arg_( arg ) {}
      virtual bool operator()( const edm::ObjectWithDict& o ) const {
	return ! (*arg_)( o );
      }
    private:
      SelectorPtr arg_;
    };
  }
}

#endif
