#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

void exitSansCUDADevices() {
  int devices = 0;
  auto status = cudaGetDeviceCount(& devices);
  if (status != cudaSuccess) {
    std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }
  if (devices == 0) {
    std::cerr << "No CUDA devices available, the test will be skipped." << "\n";
    exit(EXIT_SUCCESS);
  }
}
