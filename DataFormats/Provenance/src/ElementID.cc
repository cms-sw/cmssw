#include "DataFormats/Provenance/interface/ElementID.h"
#include <algorithm>
#include <ostream>

namespace edm {
  void ElementID::swap(ElementID& other) {
    std::swap(index_, other.index_);
    edm::swap(id_, other.id_);
  }

  bool operator<(ElementID const& lh, ElementID const& rh) {
    return lh.id() < rh.id() || (lh.id() == rh.id() && lh.index() < rh.index());
  }

  std::ostream& operator<<(std::ostream& os, ElementID const& id) {
    os << id.id() << ":" << id.index();
    return os;
  }
}  // namespace edm
