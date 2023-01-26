// C/C++ headers
#include <cstdlib>

// ROCm headers
#include <hip/hip_runtime.h>

// local headers
#include "isRocmDeviceSupported.h"

// returns EXIT_SUCCESS if at least one visible ROCm device can be used, or EXIT_FAILURE otherwise
int main() {
  int devices = 0;
  auto status = hipGetDeviceCount(&devices);
  if (status != hipSuccess) {
    return EXIT_FAILURE;
  }

  // check that at least one visible ROCm device can be used
  for (int i = 0; i < devices; ++i) {
    if (isRocmDeviceSupported(i))
      return EXIT_SUCCESS;
  }

  // no visible usable devices
  return EXIT_FAILURE;
}
