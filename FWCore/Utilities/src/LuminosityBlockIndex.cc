#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

#include <limits>

namespace edm {

  const unsigned int LuminosityBlockIndex::invalidValue_ = std::numeric_limits<unsigned int>::max();    

  LuminosityBlockIndex LuminosityBlockIndex::invalidLuminosityBlockIndex() {
    return LuminosityBlockIndex(invalidValue_);
  }
}
