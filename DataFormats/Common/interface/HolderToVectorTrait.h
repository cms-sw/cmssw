#ifndef DataFormats_Common_HolderToVectorTrait_h
#define DataFormats_Common_HolderToVectorTrait_h
#include <boost/static_assert.hpp>

namespace edm {
  namespace reftobase {
    template<typename T, typename REF>
    struct HolderToVectorTrait {
      BOOST_STATIC_ASSERT(sizeof(REF) == 0); 
    };

    template<typename REF>
    struct RefHolderToRefVectorTrait {
      BOOST_STATIC_ASSERT(sizeof(REF) == 0); 
    };

  }
}

#endif
