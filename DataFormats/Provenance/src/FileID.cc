#include "DataFormats/Provenance/interface/FileID.h"
#include <ostream>

namespace edm {
  std::ostream&
  operator<<(std::ostream& os, FileID const& id) {
    os << id.fid();
    return os;
  }
}
