#ifndef HeterogeneousCore_CUDAUtilities_CUDAEventCache_h
#define HeterogeneousCore_CUDAUtilities_CUDAEventCache_h

#include <vector>

#include <cuda_runtime.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"

class CUDAService;

namespace cudautils {
  class CUDAEventCache {
  public:
    using BareEvent = SharedEventPtr::element_type;

    CUDAEventCache();

    // Gets a (cached) CUDA event for the current device. The event
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    SharedEventPtr getCUDAEvent();

  private:
    friend class ::CUDAService;
    // intended to be called only from CUDAService destructor
    void clear();

    class Deleter {
    public:
      Deleter() = default;
      Deleter(int d) : device_{d} {}
      void operator()(cudaEvent_t event) const;

    private:
      int device_ = -1;
    };

    std::vector<edm::ReusableObjectHolder<BareEvent, Deleter>> cache_;
  };

  // Gets the global instance of a CUDAEventCache
  // This function is thread safe
  CUDAEventCache& getCUDAEventCache();
}  // namespace cudautils

#endif
