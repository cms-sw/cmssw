#ifndef HeterogeneousCore_CUDAUtilities_CUDAStreamCache_h
#define HeterogeneousCore_CUDAUtilities_CUDAStreamCache_h

#include <vector>

#include <cuda_runtime.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"

class CUDAService;

namespace cudautils {
  class CUDAStreamCache {
  public:
    using BareStream = SharedStreamPtr::element_type;

    CUDAStreamCache();

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    SharedStreamPtr getCUDAStream();

  private:
    friend class ::CUDAService;
    // intended to be called only from CUDAService destructor
    void clear();

    class Deleter {
    public:
      Deleter() = default;
      Deleter(int d) : device_{d} {}
      void operator()(cudaStream_t stream) const;

    private:
      int device_ = -1;
    };

    std::vector<edm::ReusableObjectHolder<BareStream, Deleter>> cache_;
  };

  // Gets the global instance of a CUDAStreamCache
  // This function is thread safe
  CUDAStreamCache& getCUDAStreamCache();
}  // namespace cudautils

#endif
