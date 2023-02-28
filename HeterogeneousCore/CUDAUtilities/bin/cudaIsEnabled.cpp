// C/C++ headers
#include <cstdlib>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/Common/interface/PlatformStatus.h"

// local headers
#include "isCudaDeviceSupported.h"

// returns PlatformStatus::Success if at least one visible CUDA device can be used,
// or different failure codes depending on the problem.
int main() {
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess) {
    // could not initialise the CUDA runtime
    return PlatformStatus::RuntimeNotAvailable;
  }

  // check that at least one visible CUDA device can be used
  for (int i = 0; i < devices; ++i) {
    if (isCudaDeviceSupported(i))
      return PlatformStatus::Success;
  }

  // could not find any usable CUDA devices
  return PlatformStatus::DevicesNotAvailable;
}
