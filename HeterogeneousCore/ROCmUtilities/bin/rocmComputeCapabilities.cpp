// C/C++ standard headers
#include <cstdlib>
#include <iomanip>
#include <iostream>

// ROCm headers
#include <hip/hip_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
#include "isRocmDeviceSupported.h"

int main() {
  int devices = 0;
  hipError_t status = hipGetDeviceCount(&devices);
  if (status != hipSuccess) {
    std::cerr << "rocmComputeCapabilities: " << hipGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < devices; ++i) {
    hipDeviceProp_t properties;
    hipCheck(hipGetDeviceProperties(&properties, i));
    std::stringstream arch;
    arch << "gfx" << properties.gcnArch;
    std::cout << std::setw(4) << i << "    " << std::setw(8) << arch.str() << "    " << properties.name;
    if (not isRocmDeviceSupported(i)) {
      std::cout << " (unsupported)";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
