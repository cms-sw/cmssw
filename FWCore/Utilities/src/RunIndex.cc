#include "FWCore/Utilities/interface/RunIndex.h"

#include <limits>

namespace edm {

  const unsigned int RunIndex::invalidValue_ = std::numeric_limits<unsigned int>::max();    

  RunIndex RunIndex::invalidRunIndex() {
    return RunIndex(invalidValue_);
  }
}
