// C/C++ standard headers
#include <cstdlib>
#include <iomanip>
#include <iostream>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "isCudaDeviceSupported.h"

int main() {
  int devices = 0;
  cudaError_t status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess) {
    std::cerr << "cudaComputeCapabilities: " << cudaGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < devices; ++i) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);
    std::cout << std::setw(4) << i << "    " << std::setw(2) << properties.major << "." << properties.minor << "    "
              << properties.name;
    if (not isCudaDeviceSupported(i)) {
      std::cout << " (unsupported)";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
