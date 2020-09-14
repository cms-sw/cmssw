#include "DataFormats/Common/interface/ThinnedAssociation.h"

#include <algorithm>

namespace edm {

  ThinnedAssociation::ThinnedAssociation() {}

  std::optional<unsigned int> ThinnedAssociation::getThinnedIndex(unsigned int parentIndex) const {
    auto iter = std::lower_bound(indexesIntoParent_.begin(), indexesIntoParent_.end(), parentIndex);
    if (iter != indexesIntoParent_.end() && *iter == parentIndex) {
      return iter - indexesIntoParent_.begin();
    }
    return std::nullopt;
  }
}  // namespace edm
