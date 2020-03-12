// C++ standard headers
#include <iomanip>
#include <iostream>

// CUDA headers
#include <cuda_runtime.h>

// CMSSW headers
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

int main() {
  int devices = 0;
  cudaCheck(cudaGetDeviceCount(&devices));

  for (int i = 0; i < devices; ++i) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);
    std::cout << std::setw(4) << i << "    " << std::setw(2) << properties.major << "." << properties.minor << "    "
              << properties.name << std::endl;
  }

  return 0;
}
