
#ifndef HeterogeneousCore_CUDAUtilities_interface_maxCoopBlocks_h
#define HeterogeneousCore_CUDAUtilities_interface_maxCoopBlocks_h

#include <cuda_runtime.h>

template <typename F>
inline int maxCoopBlocks(F kernel, int nthreads, int shmem, int device) {
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, nthreads, shmem);
  return deviceProp.multiProcessorCount * numBlocksPerSm;
}

#endif
