// C/C++ headers
#include <cstdlib>

// CMSSW headers
#include "HeterogeneousCore/Common/interface/PlatformStatus.h"

int main() {
  // ROCm is not available on this architecture, OS and compiler combination
  return PlatformStatus::PlatformNotAvailable;
}
