#ifndef HeterogeneousCore_AlpakaCore_interface_ScopedContext_h
#define HeterogeneousCore_AlpakaCore_interface_ScopedContext_h

#include <memory>
#include <stdexcept>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/Product.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/ContextState.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HostOnlyTask.h"
#include "HeterogeneousCore/AlpakaInterface/interface/ScopedContextFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatest {
  class TestScopedContext;
}

namespace cms::alpakatools {

  namespace impl {
    // This class is intended to be derived by other ScopedContext*, not for general use
    template <typename TQueue, typename>
    class ScopedContextBase {
    public:
      using Queue = TQueue;
      using Device = alpaka::Dev<Queue>;
      using Platform = alpaka::Pltf<Device>;

      Device device() const { return alpaka::getDev(*queue_); }

      Queue& queue() { return *queue_; }
      const std::shared_ptr<Queue>& queuePtr() const { return queue_; }

    protected:
      ScopedContextBase(ProductBase<Queue> const& data)
          : queue_{data.mayReuseQueue() ? data.queuePtr() : getQueueCache<Queue>().get(data.device())} {}

      explicit ScopedContextBase(std::shared_ptr<Queue> queue) : queue_(std::move(queue)) {}

      explicit ScopedContextBase(edm::StreamID streamID)
          : queue_{getQueueCache<Queue>().get(cms::alpakatools::chooseDevice<Platform>(streamID))} {}

    private:
      std::shared_ptr<Queue> queue_;
    };

    template <typename TQueue, typename>
    class ScopedContextGetterBase : public ScopedContextBase<TQueue> {
    public:
      using Queue = TQueue;

      template <typename T>
      const T& get(Product<Queue, T> const& data) {
        synchronizeStreams(data);
        return data.data_;
      }

      template <typename T>
      const T& get(edm::Event const& event, edm::EDGetTokenT<Product<Queue, T>> token) {
        return get(event.get(token));
      }

    protected:
      template <typename... Args>
      ScopedContextGetterBase(Args&&... args) : ScopedContextBase<Queue>{std::forward<Args>(args)...} {}

      void synchronizeStreams(ProductBase<Queue> const& data) {
        // If the product has been enqueued to a different queue, make sure that it is available before accessing it
        if (data.queue() != this->queue()) {
          // Different queues, check if the underlying device is the same
          if (data.device() != this->device()) {
            // Eventually replace with prefetch to current device (assuming unified memory works)
            // If we won't go to unified memory, need to figure out something else...
            throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
          }
          // If the data product is not yet available, synchronize the two queues
          if (not data.isAvailable()) {
            // Event not yet occurred, so need to add synchronization
            // here. Sychronization is done by making the current queue
            // wait for an event, so all subsequent work in the queue
            // will run only after the event has "occurred" (i.e. data
            // product became available).
            alpaka::wait(this->queue(), data.event());
          }
        }
      }
    };

    class ScopedContextHolderHelper {
    public:
      ScopedContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder)
          : waitingTaskHolder_{std::move(waitingTaskHolder)} {}

      template <typename F, typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
      void pushNextTask(F&& f, ContextState<TQueue> const* state) {
        replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task_with_holder(
            std::move(waitingTaskHolder_), [state, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
              func(ScopedContextTask{state, std::move(h)});
            })});
      }

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        waitingTaskHolder_ = std::move(waitingTaskHolder);
      }

      template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
      void enqueueCallback(TQueue& queue) {
        alpaka::enqueue(queue, alpaka::HostOnlyTask([holder = std::move(waitingTaskHolder_)]() {
                          // The functor is required to be const, but the original waitingTaskHolder_
                          // needs to be notified...
                          const_cast<edm::WaitingTaskWithArenaHolder&>(holder).doneWaiting(nullptr);
                        }));
      }

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
    };
  }  // namespace impl

  /**
   * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
   * - setting the current device
   * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
   * - synchronizing between queues if necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  template <typename TQueue, typename>
  class ScopedContextAcquire : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::queue;
    using ScopedContextGetterBase::queuePtr;

    /// Constructor to create a new queue (no need for context beyond acquire())
    explicit ScopedContextAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)} {}

    /// Constructor to create a new queue, and the context is needed after acquire()
    explicit ScopedContextAcquire(edm::StreamID streamID,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState<Queue>& state)
        : ScopedContextGetterBase(streamID), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    /// Constructor to (possibly) re-use a queue (no need for context beyond acquire())
    explicit ScopedContextAcquire(ProductBase<Queue> const& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)} {}

    /// Constructor to (possibly) re-use a queue, and the context is needed after acquire()
    explicit ScopedContextAcquire(ProductBase<Queue> const& data,
                                  edm::WaitingTaskWithArenaHolder waitingTaskHolder,
                                  ContextState<Queue>& state)
        : ScopedContextGetterBase(data), holderHelper_{std::move(waitingTaskHolder)}, contextState_{&state} {}

    ~ScopedContextAcquire() {
      holderHelper_.enqueueCallback(queue());
      if (contextState_) {
        contextState_->set(queuePtr());
      }
    }

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
    void throwNoState() {
      throw cms::Exception("LogicError")
          << "Calling ScopedContextAcquire::insertNextTask() requires ScopedContextAcquire to be constructed with "
             "ContextState, but that was not the case";
    }

    impl::ScopedContextHolderHelper holderHelper_;
    ContextState<Queue>* contextState_ = nullptr;
  };

  /**
   * The aim of this class is to do necessary per-event "initialization" in ExternalWork produce() or normal produce():
   * - setting the current device
   * - synchronizing between queues if necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  template <typename TQueue, typename>
  class ScopedContextProduce : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::device;
    using ScopedContextGetterBase::queue;
    using ScopedContextGetterBase::queuePtr;

    /// Constructor to re-use the queue of acquire() (ExternalWork module)
    explicit ScopedContextProduce(ContextState<Queue>& state)
        : ScopedContextGetterBase(state.releaseQueuePtr()), event_{getEventCache<Event>().get(device())} {}

    explicit ScopedContextProduce(ProductBase<Queue> const& data)
        : ScopedContextGetterBase(data), event_{getEventCache<Event>().get(device())} {}

    explicit ScopedContextProduce(edm::StreamID streamID)
        : ScopedContextGetterBase(streamID), event_{getEventCache<Event>().get(device())} {}

    /// Record the event, all asynchronous work must have been queued before the destructor
    ~ScopedContextProduce() {
      // FIXME: this may throw an execption if the underlaying call fails.
      alpaka::enqueue(queue(), *event_);
    }

    template <typename T>
    std::unique_ptr<Product<Queue, T>> wrap(T data) {
      // make_unique doesn't work because of private constructor
      return std::unique_ptr<Product<Queue, T>>(new Product<Queue, T>(queuePtr(), std::move(data)));
    }

    template <typename T, typename... Args>
    auto emplace(edm::Event& iEvent, edm::EDPutTokenT<Product<Queue, T>> token, Args&&... args) {
      return iEvent.emplace(token, queuePtr(), event_, std::forward<Args>(args)...);
    }

  private:
    friend class ::cms::alpakatest::TestScopedContext;

    explicit ScopedContextProduce(std::shared_ptr<Queue> queue)
        : ScopedContextGetterBase(std::move(queue)), event_{getEventCache<Event>().get(device())} {}

    std::shared_ptr<Event> event_;
  };

  /**
   * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
   * - setting the current device
   * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  template <typename TQueue, typename>
  class ScopedContextTask : public impl::ScopedContextBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextBase = impl::ScopedContextBase<Queue>;
    using ScopedContextBase::queue;
    using ScopedContextBase::queuePtr;

    /// Constructor to re-use the queue of acquire() (ExternalWork module)
    explicit ScopedContextTask(ContextState<Queue> const* state, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : ScopedContextBase(state->queuePtr()),  // don't move, state is re-used afterwards
          holderHelper_{std::move(waitingTaskHolder)},
          contextState_{state} {}

    ~ScopedContextTask() { holderHelper_.enqueueCallback(queue()); }

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f), contextState_);
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

  private:
    impl::ScopedContextHolderHelper holderHelper_;
    ContextState<Queue> const* contextState_;
  };

  /**
   * The aim of this class is to do necessary per-event "initialization" in analyze()
   * - setting the current device
   * - synchronizing between queues if necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  template <typename TQueue, typename>
  class ScopedContextAnalyze : public impl::ScopedContextGetterBase<TQueue> {
  public:
    using Queue = TQueue;
    using ScopedContextGetterBase = impl::ScopedContextGetterBase<Queue>;
    using ScopedContextGetterBase::queue;
    using ScopedContextGetterBase::queuePtr;

    /// Constructor to (possibly) re-use a queue
    explicit ScopedContextAnalyze(ProductBase<Queue> const& data) : ScopedContextGetterBase(data) {}
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_interface_ScopedContext_h
