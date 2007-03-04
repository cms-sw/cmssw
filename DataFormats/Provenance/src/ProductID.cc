#include "DataFormats/Provenance/interface/ProductID.h"
#include <ostream>

namespace edm {
  std::ostream&
  operator<<(std::ostream& os, ProductID const& id) {
    os << id.id_;
    return os;
  }
}
