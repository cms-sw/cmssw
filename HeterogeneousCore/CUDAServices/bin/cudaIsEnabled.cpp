// C/C++ headers
#include <cstdlib>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "isCudaDeviceSupported.h"

// returns EXIT_SUCCESS if at least one visible CUDA device can be used, or EXIT_FAILURE otherwise
int main() {
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess) {
    return EXIT_FAILURE;
  }

  // check that at least one visible CUDA device can be used
  for (int i = 0; i < devices; ++i) {
    if (isCudaDeviceSupported(i))
      return EXIT_SUCCESS;
  }

  // no visible usable devices
  return EXIT_FAILURE;
}
