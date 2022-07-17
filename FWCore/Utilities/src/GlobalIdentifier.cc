#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/Guid.h"

namespace edm {
  std::string createGlobalIdentifier(bool binary) {
    Guid guid;
    return binary ? guid.toBinary() : guid.toString();
  }

  bool isValidGlobalIdentifier(std::string const& guid) { return Guid::isValidString(guid); }
}  // namespace edm
