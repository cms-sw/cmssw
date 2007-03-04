#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include <ostream>

namespace edm {
  std::ostream&
  operator<< (std::ostream& os, FileFormatVersion const& ff) {
    os << ff.value_;
    return os;
  }
}

