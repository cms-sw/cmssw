#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAEventCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAStreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "chooseCUDADevice.h"

namespace {
  struct CallbackData {
    edm::WaitingTaskWithArenaHolder holder;
    int device;
  };

  void CUDART_CB cudaScopedContextCallback(cudaStream_t streamId, cudaError_t status, void* data) {
    std::unique_ptr<CallbackData> guard{reinterpret_cast<CallbackData*>(data)};
    edm::WaitingTaskWithArenaHolder& waitingTaskHolder = guard->holder;
    int device = guard->device;
    if (status == cudaSuccess) {
      LogTrace("CUDAScopedContext") << " GPU kernel finished (in callback) device " << device << " CUDA stream "
                                    << streamId;
      waitingTaskHolder.doneWaiting(nullptr);
    } else {
      // wrap the exception in a try-catch block to let GDB "catch throw" break on it
      try {
        auto error = cudaGetErrorName(status);
        auto message = cudaGetErrorString(status);
        throw cms::Exception("CUDAError") << "Callback of CUDA stream " << streamId << " in device " << device
                                          << " error " << error << ": " << message;
      } catch (cms::Exception&) {
        waitingTaskHolder.doneWaiting(std::current_exception());
      }
    }
  }
}  // namespace

namespace impl {
  CUDAScopedContextBase::CUDAScopedContextBase(edm::StreamID streamID)
      : currentDevice_(cudacore::chooseCUDADevice(streamID)) {
    cudaCheck(cudaSetDevice(currentDevice_));
    stream_ = cudautils::getCUDAStreamCache().getCUDAStream();
  }

  CUDAScopedContextBase::CUDAScopedContextBase(const CUDAProductBase& data)
      : currentDevice_(data.device()) {
    cudaCheck(cudaSetDevice(currentDevice_));
    if (data.mayReuseStream()) {
      stream_ = data.streamPtr();
    } else {
      stream_ = cudautils::getCUDAStreamCache().getCUDAStream();
    }
  }

  CUDAScopedContextBase::CUDAScopedContextBase(int device, cudautils::SharedStreamPtr stream)
      : currentDevice_(device), stream_(std::move(stream)) {
    cudaCheck(cudaSetDevice(currentDevice_));
  }

  ////////////////////

  void CUDAScopedContextGetterBase::synchronizeStreams(int dataDevice,
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
        auto ret = cudaStreamWaitEvent(stream(), dataEvent, 0);
        cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
      }
    }
  }

  void CUDAScopedContextHolderHelper::enqueueCallback(int device, cudaStream_t stream) {
    cudaCheck(
        cudaStreamAddCallback(stream, cudaScopedContextCallback, new CallbackData{waitingTaskHolder_, device}, 0));
  }
}  // namespace impl

////////////////////

CUDAScopedContextAcquire::~CUDAScopedContextAcquire() {
  holderHelper_.enqueueCallback(device(), stream());
  if (contextState_) {
    contextState_->set(device(), std::move(streamPtr()));
  }
}

void CUDAScopedContextAcquire::throwNoState() {
  throw cms::Exception("LogicError")
      << "Calling CUDAScopedContextAcquire::insertNextTask() requires CUDAScopedContextAcquire to be constructed with "
         "CUDAContextState, but that was not the case";
}

////////////////////

CUDAScopedContextProduce::~CUDAScopedContextProduce() {
  if (event_) {
    cudaCheck(cudaEventRecord(event_.get(), stream()));
  }
}

void CUDAScopedContextProduce::createEventIfStreamBusy() {
  if (event_) {
    return;
  }
  auto ret = cudaStreamQuery(stream());
  if (ret == cudaSuccess) {
    return;
  }
  if (ret != cudaErrorNotReady) {
    // cudaErrorNotReady indicates that the stream is busy, and thus
    // is not an error
    cudaCheck(ret);
  }

  event_ = cudautils::getCUDAEventCache().getCUDAEvent();
}

////////////////////

CUDAScopedContextTask::~CUDAScopedContextTask() { holderHelper_.enqueueCallback(device(), stream()); }
