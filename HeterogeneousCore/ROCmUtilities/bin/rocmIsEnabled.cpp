// C/C++ headers
#include <cstdlib>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/Common/interface/PlatformStatus.h"

// local headers
#include "isRocmDeviceSupported.h"

// returns PlatformStatus::Success if at least one visible ROCm device can be used,
// or different failure codes depending on the problem.
int main() {
  int devices = 0;
  auto status = hipGetDeviceCount(&devices);
  if (status != hipSuccess) {
    // could not initialise the ROCm runtime
    return PlatformStatus::RuntimeNotAvailable;
  }

  // check that at least one visible ROCm device can be used
  for (int i = 0; i < devices; ++i) {
    if (isRocmDeviceSupported(i))
      return PlatformStatus::Success;
  }

  // no usable ROCm devices were found
  return PlatformStatus::DevicesNotAvailable;
}
