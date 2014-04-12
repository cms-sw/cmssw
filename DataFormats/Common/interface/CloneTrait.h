#ifndef DataFormats_Common_CloneTrait_h
#define DataFormats_Common_CloneTrait_h
#include "DataFormats/Common/interface/CopyPolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
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

    template<typename T>
    struct CloneTrait<edm::View<T> > {
      typedef ClonePolicy<T> type;
    };

    template<typename T>
    struct CloneTrait<edm::RefToBaseVector<T> > {
      typedef CopyPolicy<RefToBase<T> > type;
    };

  }
}

#endif
