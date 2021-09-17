#include "FWCore/Utilities/interface/ProcessBlockIndex.h"

#include <limits>

namespace edm {

  const unsigned int ProcessBlockIndex::invalidValue_ = std::numeric_limits<unsigned int>::max();

  ProcessBlockIndex ProcessBlockIndex::invalidProcessBlockIndex() { return ProcessBlockIndex(invalidValue_); }
}  // namespace edm
