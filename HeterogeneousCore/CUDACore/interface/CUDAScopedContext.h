#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextState.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"

#include <cuda/api_wrappers.h>

#include <optional>

namespace cudatest {
  class TestCUDAScopedContext;
}

namespace impl {
  // This class is intended to be derived by other CUDAScopedContext*, not for general use
  class CUDAScopedContextBase {
  public:
    int device() const { return currentDevice_; }

    // cudaStream_t is a pointer to a thread-safe object, for which a
    // mutable access is needed even if the CUDAScopedContext itself
    // would be const. Therefore it is ok to return a non-const
    // pointer from a const method here.
    cudaStream_t stream() const { return stream_.get(); }
    const cudautils::SharedStreamPtr& streamPtr() const { return stream_; }

  protected:
    // The constructors set the current device device, but the device
    // is not set back to the previous value at the destructor. This
    // should be sufficient (and tiny bit faster) as all CUDA API
    // functions relying on the current device should be called from
    // the scope where this context is. The current device doesn't
    // really matter between modules (or across TBB tasks).
    explicit CUDAScopedContextBase(edm::StreamID streamID);

    explicit CUDAScopedContextBase(const CUDAProductBase& data);

    explicit CUDAScopedContextBase(int device, cudautils::SharedStreamPtr stream);

  private:
    int currentDevice_;
    cudautils::SharedStreamPtr stream_;
  };

  class CUDAScopedContextGetterBase : public CUDAScopedContextBase {
  public:
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
    template <typename... Args>
    CUDAScopedContextGetterBase(Args&&... args) : CUDAScopedContextBase(std::forward<Args>(args)...) {}

    void synchronizeStreams(int dataDevice, cudaStream_t dataStream, bool available, cudaEvent_t dataEvent);
  };

  class CUDAScopedContextHolderHelper {
  public:
    CUDAScopedContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : waitingTaskHolder_{std::move(waitingTaskHolder)} {}

    template <typename F>
    void pushNextTask(F&& f, CUDAContextState const* state);

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      waitingTaskHolder_ = std::move(waitingTaskHolder);
    }

    void enqueueCallback(int device, cudaStream_t stream);

  private:
    edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
  };
}  // namespace impl

/**
 * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
 * - setting the current device
 * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextAcquire : public impl::CUDAScopedContextGetterBase {
public:
  /// Constructor to create a new CUDA stream (no need for context beyond acquire())
  explicit CUDAScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
      : CUDAScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)} {}

  /// Constructor to create a new CUDA stream, and the context is needed after acquire()
  explicit CUDAScopedContextAcquire(edm::StreamID streamID,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                    CUDAContextState& state)
      : CUDAScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

  /// Constructor to (possibly) re-use a CUDA stream (no need for context beyond acquire())
  explicit CUDAScopedContextAcquire(const CUDAProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
      : CUDAScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)} {}

  /// Constructor to (possibly) re-use a CUDA stream, and the context is needed after acquire()
  explicit CUDAScopedContextAcquire(const CUDAProductBase& data,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                    CUDAContextState& state)
      : CUDAScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

  ~CUDAScopedContextAcquire();

  template <typename F>
  void pushNextTask(F&& f) {
    if (contextState_ == nullptr)
      throwNoState();
    holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
  }

  void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
  }

private:
  void throwNoState();

  impl::CUDAScopedContextHolderHelper holderHelper_;
  CUDAContextState* contextState_ = nullptr;
};

/**
 * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
 * - setting the current device
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextProduce : public impl::CUDAScopedContextGetterBase {
public:
  /// Constructor to create a new CUDA stream (non-ExternalWork module)
  explicit CUDAScopedContextProduce(edm::StreamID streamID) : CUDAScopedContextGetterBase(streamID) {}

  /// Constructor to (possibly) re-use a CUDA stream (non-ExternalWork module)
  explicit CUDAScopedContextProduce(const CUDAProductBase& data) : CUDAScopedContextGetterBase(data) {}

  /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
  explicit CUDAScopedContextProduce(CUDAContextState& state)
      : CUDAScopedContextGetterBase(state.device(), state.releaseStreamPtr()) {}

  ~CUDAScopedContextProduce();

  template <typename T>
  std::unique_ptr<CUDAProduct<T>> wrap(T data) {
    // make_unique doesn't work because of private constructor
    //
    // CUDAProduct<T> constructor records CUDA event to the CUDA
    // stream. The event will become "occurred" after all work queued
    // to the stream before this point has been finished.
    std::unique_ptr<CUDAProduct<T>> ret(new CUDAProduct<T>(device(), streamPtr(), std::move(data)));
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
  explicit CUDAScopedContextProduce(int device, cudautils::SharedStreamPtr stream, cudautils::SharedEventPtr event)
      : CUDAScopedContextGetterBase(device, std::move(stream)), event_{std::move(event)} {}

  void createEventIfStreamBusy();

  cudautils::SharedEventPtr event_;
};

/**
 * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
 * - setting the current device
 * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextTask : public impl::CUDAScopedContextBase {
public:
  /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
  explicit CUDAScopedContextTask(CUDAContextState const* state, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
      : CUDAScopedContextBase(state->device(), state->streamPtr()),  // don't move, state is re-used afterwards
        holderHelper_{std::move(waitingTaskHolder)},
        contextState_{state} {}

  ~CUDAScopedContextTask();

  template <typename F>
  void pushNextTask(F&& f) {
    holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
  }

  void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
  }

private:
  impl::CUDAScopedContextHolderHelper holderHelper_;
  CUDAContextState const* contextState_;
};

/**
 * The aim of this class is to do necessary per-event "initialization" in analyze()
 * - setting the current device
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
/**
 * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
 * - setting the current device
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContextAnalyze : public impl::CUDAScopedContextGetterBase {
public:
  /// Constructor to (possibly) re-use a CUDA stream
  explicit CUDAScopedContextAnalyze(const CUDAProductBase& data) : CUDAScopedContextGetterBase(data) {}
};

namespace impl {
  template <typename F>
  void CUDAScopedContextHolderHelper::pushNextTask(F&& f, CUDAContextState const* state) {
    replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{
        edm::make_waiting_task_with_holder(tbb::task::allocate_root(),
                                           std::move(waitingTaskHolder_),
                                           [state, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
                                             func(CUDAScopedContextTask{state, std::move(h)});
                                           })});
  }
}  // namespace impl

#endif
