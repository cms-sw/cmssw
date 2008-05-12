#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Utilities/interface/CRC32Calculator.h"
#include <ostream>

namespace edm {

  void BranchID::setID(std::string const& branchName) {
    cms::CRC32Calculator crc32(branchName);
    id_ = crc32.checksum();
  }

  std::ostream&
  operator<<(std::ostream& os, BranchID const& id) {
    os << id.id();
    return os;
  }
}
