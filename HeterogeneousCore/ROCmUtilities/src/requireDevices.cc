#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"

namespace cms::rocmtest {

  bool testDevices() {
    int devices = 0;
    auto status = hipGetDeviceCount(&devices);
    if (status != hipSuccess) {
      std::cerr << "Failed to initialise the ROCm runtime, the test will be skipped.\n";
      return false;
    }
    if (devices == 0) {
      std::cerr << "No ROCm devices available, the test will be skipped.\n";
      return false;
    }
    return true;
  }

  void requireDevices() {
    if (not testDevices()) {
      exit(EXIT_SUCCESS);
    }
  }

}  // namespace cms::rocmtest
