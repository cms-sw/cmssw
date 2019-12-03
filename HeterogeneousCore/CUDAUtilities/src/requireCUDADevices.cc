#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireCUDADevices.h"

bool hasCUDADevices() {
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess) {
    std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped."
              << "\n";
    return false;
  }
  if (devices == 0) {
    std::cerr << "No CUDA devices available, the test will be skipped."
              << "\n";
    return false;
  }
  return true;
}

void requireCUDADevices() {
  if (not hasCUDADevices()) {
    exit(EXIT_SUCCESS);
  }
}
