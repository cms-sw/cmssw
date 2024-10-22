#ifndef HeterogeneousCore_CUDAUtilities_EventCache_h
#define HeterogeneousCore_CUDAUtilities_EventCache_h

#include <vector>

#include <cuda_runtime.h>

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"

class CUDAService;

namespace cms {
  namespace cuda {
    class EventCache {
    public:
      using BareEvent = SharedEventPtr::element_type;

      EventCache();

      // Gets a (cached) CUDA event for the current device. The event
      // will be returned to the cache by the shared_ptr destructor. The
      // returned event is guaranteed to be in the state where all
      // captured work has completed, i.e. cudaEventQuery() == cudaSuccess.
      //
      // This function is thread safe
      SharedEventPtr get();

    private:
      friend class ::CUDAService;

      // thread safe
      SharedEventPtr makeOrGet(int dev);

      // not thread safe, intended to be called only from CUDAService destructor
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

    // Gets the global instance of a EventCache
    // This function is thread safe
    EventCache& getEventCache();
  }  // namespace cuda
}  // namespace cms

#endif
