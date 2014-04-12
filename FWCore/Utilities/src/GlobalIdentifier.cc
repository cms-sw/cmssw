#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "Guid.h"

namespace edm {
  std::string
  createGlobalIdentifier() {
    Guid guid;
    return guid.toString();
  }
}
