#include <cassert>
#include <iostream>

#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaDeviceCount.h"

__global__ void testBinaryFind() {
  int data[] = {1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6};

  auto lower = cuda_std::lower_bound(data, data + 13, 4);
  auto upper = cuda_std::upper_bound(data, data + 12, 4);

  assert(3 == upper - lower);

  // classic binary search, returning a value only if it is present

  constexpr int data2[] = {1, 2, 4, 6, 9, 10};

  assert(data2 + 2 == cuda_std::binary_find(data2, data2 + 6, 4));
  assert(data2 + 6 == cuda_std::binary_find(data2, data2 + 6, 5));
}

void wrapper() {
  if (cudautils::cudaDeviceCount() == 0) {
    std::cerr << "No CUDA devices on this system"
              << "\n";
    exit(EXIT_FAILURE);
  }

  cudautils::launch(testBinaryFind, {32, 64});
}

int main() {
  exitSansCUDADevices();

  wrapper();
}
