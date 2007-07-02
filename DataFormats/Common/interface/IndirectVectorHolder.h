#ifndef Common_IndirectVectorHolder_h
#define Common_IndirectVectorHolder_h
#include "DataFormats/Common/interface/BaseVectorHolder.h"

namespace edm {
  namespace reftobase {

    template <class T, class TRefVector>
    class IndirectVectorHolder : public BaseVectorHolder<T> {
    };

  }
}

#endif
