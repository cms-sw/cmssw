#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseCUDADevice.h"

namespace impl {
  CUDAScopedContextBase::CUDAScopedContextBase(edm::StreamID streamID):
    currentDevice_(cudacore::chooseCUDADevice(streamID)),
    setDeviceForThisScope_(currentDevice_)
  {
    edm::Service<CUDAService> cs;
    stream_ = cs->getCUDAStream();
  }

  CUDAScopedContextBase::CUDAScopedContextBase(const CUDAProductBase& data):
    currentDevice_(data.device()),
    setDeviceForThisScope_(currentDevice_)
  {
    if(data.mayReuseStream()) {
      stream_ = data.streamPtr();
    }
    else {
      edm::Service<CUDAService> cs;
      stream_ = cs->getCUDAStream();
    }
  }

  CUDAScopedContextBase::CUDAScopedContextBase(int device, std::shared_ptr<cuda::stream_t<>> stream):
    currentDevice_(device),
    setDeviceForThisScope_(device),
    stream_(std::move(stream))
  {}

  ////////////////////

  void CUDAScopedContextGetterBase::synchronizeStreams(int dataDevice, const cuda::stream_t<>& dataStream, bool available, const cuda::event_t *dataEvent) {
    if(dataDevice != device()) {
      // Eventually replace with prefetch to current device (assuming unified memory works)
      // If we won't go to unified memory, need to figure out something else...
      throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
    }

    if(dataStream.id() != stream().id()) {
      // Different streams, need to synchronize
      if(not available) {
        // Event not yet occurred, so need to add synchronization
        // here. Sychronization is done by making the CUDA stream to
        // wait for an event, so all subsequent work in the stream
        // will run only after the event has "occurred" (i.e. data
        // product became available).
        auto ret = cudaStreamWaitEvent(stream().id(), dataEvent->id(), 0);
        cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
      }
    }
  }

  void CUDAScopedContextHolderHelper::enqueueCallback(int device, cuda::stream_t<>& stream) {
    stream.enqueue.callback([device,
                             waitingTaskHolder=waitingTaskHolder_]
                            (cuda::stream::id_t streamId, cuda::status_t status) mutable {
                              if(cuda::is_success(status)) {
                                LogTrace("CUDAScopedContext") << " GPU kernel finished (in callback) device " << device << " CUDA stream " << streamId;
                                waitingTaskHolder.doneWaiting(nullptr);
                              }
                              else {
                                // wrap the exception in a try-catch block to let GDB "catch throw" break on it
                                try {
                                  auto error = cudaGetErrorName(status);
                                  auto message = cudaGetErrorString(status);
                                  throw cms::Exception("CUDAError") << "Callback of CUDA stream " << streamId << " in device " << device << " error " << error << ": " << message;
                                } catch(cms::Exception&) {
                                  waitingTaskHolder.doneWaiting(std::current_exception());
                                }
                              }
                            });
  }
}

////////////////////

CUDAScopedContextAcquire::~CUDAScopedContextAcquire() {
  holderHelper_.enqueueCallback(device(), stream());
  if(contextState_) {
    contextState_->set(device(), std::move(streamPtr()));
  }
}

void CUDAScopedContextAcquire::throwNoState() {
  throw cms::Exception("LogicError") << "Calling CUDAScopedContextAcquire::insertNextTask() requires CUDAScopedContextAcquire to be constructed with CUDAContextState, but that was not the case";
}

////////////////////

CUDAScopedContextProduce::~CUDAScopedContextProduce() {
  if(event_) {
    event_->record(stream().id());
  }
}

void CUDAScopedContextProduce::createEventIfStreamBusy() {
  if(event_ or stream().is_clear()) {
    return;
  }
  edm::Service<CUDAService> cs;
  event_ = cs->getCUDAEvent();
}

////////////////////

CUDAScopedContextTask::~CUDAScopedContextTask() {
  holderHelper_.enqueueCallback(device(), stream());
}
