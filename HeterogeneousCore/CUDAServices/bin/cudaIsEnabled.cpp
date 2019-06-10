#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

int main() {
  int devices = 0;
  auto status = cudaGetDeviceCount(& devices);
  if (status != cudaSuccess) {
    return EXIT_FAILURE;
  }

  int minimumMajor = 6; // min minor is implicitly 0

  // This approach (requiring all devices are supported) is rather
  // conservative. In principle we could consider just dropping the
  // unsupported devices. Currently that would be easiest to achieve
  // in CUDAService though.
  for (int i = 0; i < devices; ++i) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);

    if(properties.major < minimumMajor) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
