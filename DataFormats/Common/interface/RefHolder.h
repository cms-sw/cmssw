#ifndef DataFormats_Common_RefHolder_h
#define DataFormats_Common_RefHolder_h
#include "DataFormats/Common/interface/RefHolder_.h"

#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/HolderToVectorTrait.h"
#include <memory>

namespace edm {
  namespace reftobase {
    template <class REF>
    std::auto_ptr<RefVectorHolderBase> RefHolder<REF>::makeVectorHolder() const {
      typedef typename RefHolderToRefVectorTrait<REF>::type helper;
      return helper::makeVectorHolder();
    }
  }
}

#include "DataFormats/Common/interface/RefKeyTrait.h"

namespace edm {
  namespace reftobase {
    template <class REF>
    size_t
    RefHolder<REF>::key() const 
    {
      typedef typename RefKeyTrait<REF>::type helper;
      return helper::key( ref_ );
    }

  }
}

#endif
