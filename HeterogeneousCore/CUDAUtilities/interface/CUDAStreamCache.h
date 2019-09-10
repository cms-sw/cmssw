#ifndef HeterogeneousCore_CUDAUtilities_CUDAStreamCache_h
#define HeterogeneousCore_CUDAUtilities_CUDAStreamCache_h

#include <memory>

#include <cuda/api_wrappers.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

class CUDAService;

namespace cudautils {
  class CUDAStreamCache {
  public:
    CUDAStreamCache();

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<cuda::stream_t<>> getCUDAStream();

  private:
    friend class ::CUDAService;
    // intended to be called only from CUDAService destructor
    void clear();

    std::vector<edm::ReusableObjectHolder<cuda::stream_t<> > > cache_;
  };

  // Gets the global instance of a CUDAStreamCache
  // This function is thread safe
  CUDAStreamCache& getCUDAStreamCache();
}

#endif
