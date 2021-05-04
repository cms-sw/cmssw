#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

/**
 * This file tests that the assert() and #include ".../cuda_assert.h"
 * compiles and runs when compiled with and without -DGPU_DEBUG (see
 * also BuildFile.xml).
 */

__global__ void testIt(int one) { assert(one == 1); }

int main(int argc, char* argv[]) {
  cms::cudatest::requireDevices();

  testIt<<<1, 1>>>(argc);
  cudaDeviceSynchronize();

  return (argc == 1) ? EXIT_SUCCESS : EXIT_FAILURE;
}
