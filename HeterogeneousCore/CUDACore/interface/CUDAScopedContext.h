#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextToken.h"

#include <cuda/api_wrappers.h>

#include <optional>

namespace cudatest {
  class TestCUDAScopedContext;
}

/**
 * The aim of this class is to do necessary per-event "initialization":
 * - setting the current device
 * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContext {
public:
  explicit CUDAScopedContext(edm::StreamID streamID);

  explicit CUDAScopedContext(CUDAContextToken&& token):
    currentDevice_(token.device()),
    setDeviceForThisScope_(currentDevice_),
    stream_(std::move(token.streamPtr()))
  {}

  explicit CUDAScopedContext(const CUDAProductBase& data);

  explicit CUDAScopedContext(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContext(streamID)
  {
    waitingTaskHolder_ = std::move(waitingTaskHolder);
  }

  explicit CUDAScopedContext(const CUDAProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContext(data)
  {
    waitingTaskHolder_ = std::move(waitingTaskHolder);
  }

  ~CUDAScopedContext();

  int device() const { return currentDevice_; }

  cuda::stream_t<>& stream() { return *stream_; }
  const cuda::stream_t<>& stream() const { return *stream_; }
  const std::shared_ptr<cuda::stream_t<>>& streamPtr() const { return stream_; }

  CUDAContextToken toToken() {
    return CUDAContextToken(currentDevice_, stream_);
  }

  template <typename T>
  const T& get(const CUDAProduct<T>& data) {
    synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
    return data.data_;
  }

  template <typename T>
  const T& get(const edm::Event& iEvent, edm::EDGetTokenT<CUDAProduct<T>> token) {
    return get(iEvent.get(token));
  }

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
  explicit CUDAScopedContext(int device, std::unique_ptr<cuda::stream_t<>> stream, std::unique_ptr<cuda::event_t> event);

  void createEventIfStreamBusy();
  void synchronizeStreams(int dataDevice, const cuda::stream_t<>& dataStream, bool available, const cuda::event_t *dataEvent);

  int currentDevice_;
  std::optional<edm::WaitingTaskWithArenaHolder> waitingTaskHolder_;
  cuda::device::current::scoped_override_t<> setDeviceForThisScope_;
  std::shared_ptr<cuda::stream_t<>> stream_;
  std::shared_ptr<cuda::event_t> event_;
};

#endif
