#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextState.h"

#include <cuda/api_wrappers.h>

#include <optional>

namespace cudatest {
  class TestCUDAScopedContext;
}

// This class is intended to be derived by other CUDAScopedContext*, not for general use
class CUDAScopedContextBase {
public:
  int device() const { return currentDevice_; }

  cuda::stream_t<>& stream() { return *stream_; }
  const cuda::stream_t<>& stream() const { return *stream_; }
  const std::shared_ptr<cuda::stream_t<>>& streamPtr() const { return stream_; }

  template <typename T>
  const T& get(const CUDAProduct<T>& data) {
    synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
    return data.data_;
  }

  template <typename T>
  const T& get(const edm::Event& iEvent, edm::EDGetTokenT<CUDAProduct<T>> token) {
    return get(iEvent.get(token));
  }

protected:
  explicit CUDAScopedContextBase(edm::StreamID streamID);

  explicit CUDAScopedContextBase(const CUDAProductBase& data);

  explicit CUDAScopedContextBase(int device, std::shared_ptr<cuda::stream_t<>> stream);

  void synchronizeStreams(int dataDevice, const cuda::stream_t<>& dataStream, bool available, const cuda::event_t *dataEvent);

  std::shared_ptr<cuda::stream_t<>>& streamPtr() { return stream_; }

private:
  int currentDevice_;
  cuda::device::current::scoped_override_t<> setDeviceForThisScope_;
  std::shared_ptr<cuda::stream_t<>> stream_;
};

/**
 * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
 * - setting the current device
 * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextAcquire: public CUDAScopedContextBase {
public:
  /// Constructor to create a new CUDA stream (no need for context beyond acquire())
  explicit CUDAScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContextBase(streamID),
    waitingTaskHolder_{std::move(waitingTaskHolder)}
  {}

  /// Constructor to create a new CUDA stream, and the context is needed after acquire()
  explicit CUDAScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, CUDAContextState& state):
    CUDAScopedContextBase(streamID),
    waitingTaskHolder_{std::move(waitingTaskHolder)},
    contextState_{&state}
  {}

  /// Constructor to (possibly) re-use a CUDA stream (no need for context beyond acquire())
  explicit CUDAScopedContextAcquire(const CUDAProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContextBase(data),
    waitingTaskHolder_{std::move(waitingTaskHolder)}
  {}

  /// Constructor to (possibly) re-use a CUDA stream, and the context is needed after acquire()
  explicit CUDAScopedContextAcquire(const CUDAProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder, CUDAContextState& state):
    CUDAScopedContextBase(data),
    waitingTaskHolder_{std::move(waitingTaskHolder)},
    contextState_{&state}
  {}

  ~CUDAScopedContextAcquire();

private:
  edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
  CUDAContextState *contextState_ = nullptr;
};

/**
 * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
 * - setting the current device
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextProduce: public CUDAScopedContextBase {
public:
  /// Constructor to create a new CUDA stream (non-ExternalWork module)
  explicit CUDAScopedContextProduce(edm::StreamID streamID):
    CUDAScopedContextBase(streamID)
  {}

  /// Constructor to (possibly) re-use a CUDA stream (non-ExternalWork module)
  explicit CUDAScopedContextProduce(const CUDAProductBase& data):
    CUDAScopedContextBase(data)
  {}

  /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
  explicit CUDAScopedContextProduce(CUDAContextState& token):
    CUDAScopedContextBase(token.device(), std::move(token.streamPtr()))
  {}

  ~CUDAScopedContextProduce();

  template <typename T>
  std::unique_ptr<CUDAProduct<T> > wrap(T data) {
    // make_unique doesn't work because of private constructor
    //
    // CUDAProduct<T> constructor records CUDA event to the CUDA
    // stream. The event will become "occurred" after all work queued
    // to the stream before this point has been finished.
    std::unique_ptr<CUDAProduct<T> > ret(new CUDAProduct<T>(device(), streamPtr(), std::move(data)));
    createEventIfStreamBusy();
    ret->setEvent(event_);
    return ret;
  }

  template <typename T, typename... Args>
  auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
    auto ret = iEvent.emplace(token, device(), streamPtr(), std::forward<Args>(args)...);
    createEventIfStreamBusy();
    const_cast<T&>(*ret).setEvent(event_);
    return ret;
  }

private:
  friend class cudatest::TestCUDAScopedContext;

  // This construcor is only meant for testing
  explicit CUDAScopedContextProduce(int device, std::unique_ptr<cuda::stream_t<>> stream, std::unique_ptr<cuda::event_t> event):
    CUDAScopedContextBase(device, std::move(stream)),
    event_{std::move(event)}
  {}

  void createEventIfStreamBusy();

  std::shared_ptr<cuda::event_t> event_;
};

#endif
