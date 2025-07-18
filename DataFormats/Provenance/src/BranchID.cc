#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Utilities/interface/calculateCRC32.h"
#include <ostream>

namespace edm {

  BranchID::value_type BranchID::toID(std::string const& branchName) { return cms::calculateCRC32(branchName); }

  std::ostream& operator<<(std::ostream& os, BranchID const& id) {
    os << id.id();
    return os;
  }
}  // namespace edm
