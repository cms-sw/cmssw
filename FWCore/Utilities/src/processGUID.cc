#include "FWCore/Utilities/interface/processGUID.h"

namespace edm {
  Guid const& processGUID() {
    static Guid const guid;
    return guid;
  }
}  // namespace edm
