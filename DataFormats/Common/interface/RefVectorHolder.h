#ifndef Common_RefVectorHolder_h
#define Common_RefVectorHolder_h
#include "DataFormats/Common/interface/RefVectorHolderBase.h"

namespace edm {
  namespace reftobase {
    template<typename REFV>
    class RefVectorHolder : public RefVectorHolderBase  {
    };
  }
}

#endif
