#ifndef HeterogeneousCore_CUDACore_ScopedContext_h
#define HeterogeneousCore_CUDACore_ScopedContext_h

#include <optional>

#include "CUDADataFormats/Common/interface/Product.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAUtilities/interface/EventCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedEventPtr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"

namespace cms {
  namespace cudatest {
    class TestScopedContext;
  }

  namespace cuda {

    namespace impl {
      // This class is intended to be derived by other ScopedContext*, not for general use
      class ScopedContextBase {
      public:
        int device() const { return currentDevice_; }

        // cudaStream_t is a pointer to a thread-safe object, for which a
        // mutable access is needed even if the ScopedContext itself
        // would be const. Therefore it is ok to return a non-const
        // pointer from a const method here.
        cudaStream_t stream() const { return stream_.get(); }
        const SharedStreamPtr& streamPtr() const { return stream_; }

      protected:
        // The constructors set the current device, but the device
        // is not set back to the previous value at the destructor. This
        // should be sufficient (and tiny bit faster) as all CUDA API
        // functions relying on the current device should be called from
        // the scope where this context is. The current device doesn't
        // really matter between modules (or across TBB tasks).
        explicit ScopedContextBase(edm::StreamID streamID);

        explicit ScopedContextBase(const ProductBase& data);

        explicit ScopedContextBase(int device, SharedStreamPtr stream);

      private:
        int currentDevice_;
        SharedStreamPtr stream_;
      };

      class ScopedContextGetterBase : public ScopedContextBase {
      public:
        template <typename T>
        const T& get(const Product<T>& data) {
          synchronizeStreams(data.device(), data.stream(), data.isAvailable(), data.event());
          return data.data_;
        }

        template <typename T>
        const T& get(const edm::Event& iEvent, edm::EDGetTokenT<Product<T>> token) {
          return get(iEvent.get(token));
        }

      protected:
        template <typename... Args>
        ScopedContextGetterBase(Args&&... args) : ScopedContextBase(std::forward<Args>(args)...) {}

        void synchronizeStreams(int dataDevice, cudaStream_t dataStream, bool available, cudaEvent_t dataEvent);
      };

      class ScopedContextHolderHelper {
      public:
        ScopedContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder)
            : waitingTaskHolder_{std::move(waitingTaskHolder)} {}

        template <typename F>
        void pushNextTask(F&& f, ContextState const* state);

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
    class ScopedContextAcquire : public impl::ScopedContextGetterBase {
    public:
      /// Constructor to create a new CUDA stream (no need for context beyond acquire())
      explicit ScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)} {}

      /// Constructor to create a new CUDA stream, and the context is needed after acquire()
      explicit ScopedContextAcquire(edm::StreamID streamID,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                    ContextState& state)
          : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

      /// Constructor to (possibly) re-use a CUDA stream (no need for context beyond acquire())
      explicit ScopedContextAcquire(const ProductBase& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)} {}

      /// Constructor to (possibly) re-use a CUDA stream, and the context is needed after acquire()
      explicit ScopedContextAcquire(const ProductBase& data,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                    ContextState& state)
          : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

      ~ScopedContextAcquire();

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

      impl::ScopedContextHolderHelper holderHelper_;
      ContextState* contextState_ = nullptr;
    };

    /**
     * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
    class ScopedContextProduce : public impl::ScopedContextGetterBase {
    public:
      /// Constructor to create a new CUDA stream (non-ExternalWork module)
      explicit ScopedContextProduce(edm::StreamID streamID) : ScopedContextGetterBase(streamID) {}

      /// Constructor to (possibly) re-use a CUDA stream (non-ExternalWork module)
      explicit ScopedContextProduce(const ProductBase& data) : ScopedContextGetterBase(data) {}

      /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
      explicit ScopedContextProduce(ContextState& state)
          : ScopedContextGetterBase(state.device(), state.releaseStreamPtr()) {}

      /// Record the CUDA event, all asynchronous work must have been queued before the destructor
      ~ScopedContextProduce();

      template <typename T>
      std::unique_ptr<Product<T>> wrap(T data) {
        // make_unique doesn't work because of private constructor
        return std::unique_ptr<Product<T>>(new Product<T>(device(), streamPtr(), event_, std::move(data)));
      }

      template <typename T, typename... Args>
      auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
        return iEvent.emplace(token, device(), streamPtr(), event_, std::forward<Args>(args)...);
      }

    private:
      friend class cudatest::TestScopedContext;

      // This construcor is only meant for testing
      explicit ScopedContextProduce(int device, SharedStreamPtr stream, SharedEventPtr event)
          : ScopedContextGetterBase(device, std::move(stream)), event_{std::move(event)} {}

      // create the CUDA Event upfront to catch possible errors from its creation
      SharedEventPtr event_ = getEventCache().get();
    };

    /**
     * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
     * - setting the current device
     * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
    class ScopedContextTask : public impl::ScopedContextBase {
    public:
      /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
      explicit ScopedContextTask(ContextState const* state, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : ScopedContextBase(state->device(), state->streamPtr()),  // don't move, state is re-used afterwards
            holderHelper_{std::move(waitingTaskHolder)},
            contextState_{state} {}

      ~ScopedContextTask();

      template <typename F>
      void pushNextTask(F&& f) {
        holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
      }

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
      }

    private:
      impl::ScopedContextHolderHelper holderHelper_;
      ContextState const* contextState_;
    };

    /**
     * The aim of this class is to do necessary per-event "initialization" in analyze()
     * - setting the current device
     * - synchronizing between CUDA streams if necessary
     * and enforce that those get done in a proper way in RAII fashion.
     */
    class ScopedContextAnalyze : public impl::ScopedContextGetterBase {
    public:
      /// Constructor to (possibly) re-use a CUDA stream
      explicit ScopedContextAnalyze(const ProductBase& data) : ScopedContextGetterBase(data) {}
    };

    namespace impl {
      template <typename F>
      void ScopedContextHolderHelper::pushNextTask(F&& f, ContextState const* state) {
        replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{
            edm::make_waiting_task_with_holder(tbb::task::allocate_root(),
                                               std::move(waitingTaskHolder_),
                                               [state, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
                                                 func(ScopedContextTask{state, std::move(h)});
                                               })});
      }
    }  // namespace impl
  }    // namespace cuda
}  // namespace cms

#endif
