#include "DataFormats/Provenance/interface/BranchID.h"
#include <ostream>

namespace edm {
  // These will be replaced with crc32
  BranchID::BranchID() : id_(ID()) {}

  // These will be replaced with crc32
  BranchID::BranchID(std::string const& str) : id_(str) {}

  std::ostream&
  operator<<(std::ostream& os, BranchID const& id) {
    os << id.id();
    return os;
  }
}
