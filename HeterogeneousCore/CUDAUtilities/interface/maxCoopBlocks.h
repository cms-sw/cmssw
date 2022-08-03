
#ifndef HeterogeneousCore_CUDAUtilities_interface_maxCoopBlocks_h
#define HeterogeneousCore_CUDAUtilities_interface_maxCoopBlocks_h

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>

template <typename F>
inline int maxCoopBlocks(F kernel, int nthreads, int shmem, int device, int redFact = 10) {
// #define GET_COOP_RED_FACT_FROM_ENV

// to drive performance assessment by envvar
#ifdef GET_COOP_RED_FACT_FROM_ENV
  auto env = std::getenv("COOP_RED_FACT");
  int redFactFromEnv = env ? atoi(env) : 0;
  if (redFactFromEnv != 0)
    redFact = redFactFromEnv;
#endif

  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, nthreads, shmem);
  int mxblocks = deviceProp.multiProcessorCount * numBlocksPerSm;
  // reduce number of blocks to account for multiple CPU threads
  return std::max(1, mxblocks / redFact);
}

#endif
