#ifndef DataFormats_Common_MapRefViewTrait_h
#define DataFormats_Common_MapRefViewTrait_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include <map>

namespace edm {
  namespace helper {
    template<typename C>
    struct MapRefViewTrait {
      typedef Ref<C> ref_type;
      typedef RefProd<C> refprod_type;
    }; 
    
    template<typename T>
    struct MapRefViewTrait<View<T> > {
      typedef RefToBase<T> ref_type;
      typedef RefToBaseProd<T> refprod_type;
    }; 
  }
}

#endif
