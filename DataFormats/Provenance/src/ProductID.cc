#include "DataFormats/Provenance/interface/ProductID.h"
#include <ostream>
#include <algorithm>

namespace edm {
  std::ostream&
  operator<<(std::ostream& os, ProductID const& id) {
    os << id.processIndex() << ":" << id.productIndex();
    return os;
  }

  bool operator<(ProductID const& lh, ProductID const& rh) {
    return lh.processIndex() < rh.processIndex() ||
      (lh.processIndex() == rh.processIndex() && lh.productIndex() < rh.productIndex());
  }

  void ProductID::swap(ProductID& other) {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    std::swap(oldID_, other.oldID_);
  }
}
