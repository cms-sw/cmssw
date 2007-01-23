#ifndef Common_CloneTrait
#define Common_CloneTrait
#include "DataFormats/Common/interface/CopyPolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <vector>

namespace edm {
  namespace clonehelper {
    template<typename T> struct CloneTrait;

    template<typename T>
    struct CloneTrait<std::vector<T> > {
      typedef CopyPolicy<T> type;
    };

    template<typename T>
    struct CloneTrait<edm::OwnVector<T> > {
      typedef ClonePolicy<T> type;
    };

  }
}

#endif
