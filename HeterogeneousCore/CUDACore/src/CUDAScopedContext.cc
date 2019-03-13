#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseCUDADevice.h"


CUDAScopedContext::CUDAScopedContext(edm::StreamID streamID):
  currentDevice_(cudacore::chooseCUDADevice(streamID)),
  setDeviceForThisScope_(currentDevice_)
{
  edm::Service<CUDAService> cs;
  stream_ = cs->getCUDAStream();
}

CUDAScopedContext::CUDAScopedContext(int device, std::unique_ptr<cuda::stream_t<>> stream):
  currentDevice_(device),
  setDeviceForThisScope_(device),
  stream_(std::move(stream))
{}

CUDAScopedContext::~CUDAScopedContext() {
  if(waitingTaskHolder_.has_value()) {
    stream_->enqueue.callback([device=currentDevice_,
                               waitingTaskHolder=*waitingTaskHolder_]
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

void CUDAScopedContext::synchronizeStreams(int dataDevice, const cuda::stream_t<>& dataStream, const cuda::event_t& dataEvent) {
  if(dataDevice != currentDevice_) {
    // Eventually replace with prefetch to current device (assuming unified memory works)
    // If we won't go to unified memory, need to figure out something else...
    throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
  }

  if(dataStream.id() != stream_->id()) {
    // Different streams, need to synchronize
    if(!dataEvent.has_occurred()) {
      // Event not yet occurred, so need to add synchronization
      // here. Sychronization is done by making the CUDA stream to
      // wait for an event, so all subsequent work in the stream
      // will run only after the event has "occurred" (i.e. data
      // product became available).
      auto ret = cudaStreamWaitEvent(stream_->id(), dataEvent.id(), 0);
      cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
    }
  }
}
