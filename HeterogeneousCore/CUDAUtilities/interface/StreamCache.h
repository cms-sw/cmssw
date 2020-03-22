#ifndef HeterogeneousCore_CUDAUtilities_StreamCache_h
#define HeterogeneousCore_CUDAUtilities_StreamCache_h

#include <vector>

#include <cuda_runtime.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"

class CUDAService;

namespace cms {
  namespace cuda {
    class StreamCache {
    public:
      using BareStream = SharedStreamPtr::element_type;

      StreamCache();

      // Gets a (cached) CUDA stream for the current device. The stream
      // will be returned to the cache by the shared_ptr destructor.
      // This function is thread safe
      SharedStreamPtr get();

    private:
      friend class ::CUDAService;
      // not thread safe, intended to be called only from CUDAService destructor
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

    // Gets the global instance of a StreamCache
    // This function is thread safe
    StreamCache& getStreamCache();
  }  // namespace cuda
}  // namespace cms

#endif
