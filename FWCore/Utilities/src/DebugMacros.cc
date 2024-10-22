
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <cstdlib>

namespace edm {

  debugvalue::debugvalue() : cvalue_(std::getenv("PROC_DEBUG")), value_(cvalue_ == nullptr ? 0 : atoi(cvalue_)) {}

  debugvalue debugit;

}  // namespace edm
