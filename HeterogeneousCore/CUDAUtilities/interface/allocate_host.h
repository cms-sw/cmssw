#ifndef HeterogeneousCore_CUDAUtilities_allocate_host_h
#define HeterogeneousCore_CUDAUtilities_allocate_host_h

#include <cuda/api_wrappers.h>

namespace cudautils {
  // Allocate pinned host memory (to be called from unique_ptr)
  void *allocate_host(size_t nbytes, cuda::stream_t<>& stream);

  // Free pinned host memory (to be called from unique_ptr)
  void free_host(void *ptr);
}

#endif
