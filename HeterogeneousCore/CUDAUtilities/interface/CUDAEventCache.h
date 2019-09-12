#ifndef HeterogeneousCore_CUDAUtilities_CUDAEventCache_h
#define HeterogeneousCore_CUDAUtilities_CUDAEventCache_h

#include <memory>

#include <cuda/api_wrappers.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

class CUDAService;

namespace cudautils {
  class CUDAEventCache {
  public:
    CUDAEventCache();

    // Gets a (cached) CUDA event for the current device. The event
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<cuda::event_t> getCUDAEvent();

  private:
    friend class ::CUDAService;
    // intended to be called only from CUDAService destructor
    void clear();

    std::vector<edm::ReusableObjectHolder<cuda::event_t>> cache_;
  };

  // Gets the global instance of a CUDAEventCache
  // This function is thread safe
  CUDAEventCache& getCUDAEventCache();
}  // namespace cudautils

#endif
