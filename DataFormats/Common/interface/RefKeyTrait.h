#ifndef DataFormats_Common_RefKeyTrait_h
#define DataFormats_Common_RefKeyTrait_h
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  template <typename C> class RefProd;
  namespace reftobase {

    struct RefKey {
      template<typename REF>
      static size_t key( const REF & r ) {
	return r.key();
      }
    };
    
    struct RefProdKey {
      template<typename REF>
      static size_t key( const REF & r ) {
	Exception::throwThis(errors::InvalidReference,
	  "attempting get key from a RefToBase containing a RefProd.\n"
	  "You can use key only with RefToBase containing a Ref.");
        return 0;
      }
    };
    
    template<typename REF>
    struct RefKeyTrait {
      typedef RefKey type;
    };

    template<typename C>
    struct RefKeyTrait<RefProd<C> > {
      typedef RefProdKey type;
    };
  }
}

#endif
