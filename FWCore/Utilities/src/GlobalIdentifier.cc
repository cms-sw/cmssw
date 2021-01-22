#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "Guid.h"

namespace edm {
  std::string createGlobalIdentifier(bool binary) {
    Guid guid;
    return binary ? guid.toBinary() : guid.toString();
  }
}  // namespace edm
