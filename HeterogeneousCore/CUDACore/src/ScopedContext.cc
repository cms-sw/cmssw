#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "chooseDevice.h"

namespace cms::cuda {
  namespace impl {
    ScopedContextBase::ScopedContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
      cudaCheck(cudaSetDevice(currentDevice_));
      stream_ = getStreamCache().get();
    }

    ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
      cudaCheck(cudaSetDevice(currentDevice_));
      if (data.mayReuseStream()) {
        stream_ = data.streamPtr();
      } else {
        stream_ = getStreamCache().get();
      }
    }

    ScopedContextBase::ScopedContextBase(int device, SharedStreamPtr stream)
        : currentDevice_(device), stream_(std::move(stream)) {
      cudaCheck(cudaSetDevice(currentDevice_));
    }

    ////////////////////

    void ScopedContextGetterBase::synchronizeStreams(int dataDevice,
                                                     cudaStream_t dataStream,
                                                     bool available,
                                                     cudaEvent_t dataEvent) {
      if (dataDevice != device()) {
        // Eventually replace with prefetch to current device (assuming unified memory works)
        // If we won't go to unified memory, need to figure out something else...
        throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
      }

      if (dataStream != stream()) {
        // Different streams, need to synchronize
        if (not available) {
          // Event not yet occurred, so need to add synchronization
          // here. Sychronization is done by making the CUDA stream to
          // wait for an event, so all subsequent work in the stream
          // will run only after the event has "occurred" (i.e. data
          // product became available).
          cudaCheck(cudaStreamWaitEvent(stream(), dataEvent, 0), "Failed to make a stream to wait for an event");
        }
      }
    }

    void ScopedContextHolderHelper::enqueueCallback(int device, cudaStream_t stream) {
      edm::Service<edm::Async> async;
      SharedEventPtr event = getEventCache().get();
      cudaCheck(cudaEventRecord(event.get(), stream));
      async->runAsync(
          std::move(waitingTaskHolder_),
          [event = std::move(event)]() mutable { cudaCheck(cudaEventSynchronize(event.get())); },
          []() { return "Enqueued by cms::cuda::ScopedContextHolderHelper::enqueueCallback()"; });
    }
  }  // namespace impl

  ////////////////////

  ScopedContextAcquire::~ScopedContextAcquire() noexcept(false) {
    holderHelper_.enqueueCallback(device(), stream());
    if (contextState_) {
      contextState_->set(device(), streamPtr());
    }
  }

  void ScopedContextAcquire::throwNoState() {
    throw cms::Exception("LogicError")
        << "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
           "ContextState, but that was not the case";
  }

  ////////////////////

  ScopedContextProduce::~ScopedContextProduce() {
    // Intentionally not checking the return value to avoid throwing
    // exceptions. If this call would fail, we should get failures
    // elsewhere as well.
    cudaEventRecord(event_.get(), stream());
  }
}  // namespace cms::cuda
