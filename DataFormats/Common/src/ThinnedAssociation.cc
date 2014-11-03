#include "DataFormats/Common/interface/ThinnedAssociation.h"

#include <algorithm>

namespace edm {

  ThinnedAssociation::ThinnedAssociation() {
  }

  bool
  ThinnedAssociation::hasParentIndex(unsigned int parentIndex,
                                     unsigned int& thinnedIndex) const {
    auto iter = std::lower_bound(indexesIntoParent_.begin(), indexesIntoParent_.end(), parentIndex);
    if(iter != indexesIntoParent_.end() && *iter == parentIndex) {
      thinnedIndex = iter - indexesIntoParent_.begin();
      return true;
    }
    return false;
  }
}
