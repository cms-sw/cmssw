#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <cuda/api_wrappers.h>

namespace cudautils {
  // Allocate device memory
  void *allocate_device(int dev, size_t nbytes, cuda::stream_t<> &stream);

  // Free device memory (to be called from unique_ptr)
  void free_device(int device, void *ptr);
}  // namespace cudautils

#endif
