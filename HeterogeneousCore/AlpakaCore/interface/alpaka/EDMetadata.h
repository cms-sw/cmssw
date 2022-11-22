#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadata_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadata_h

#include <atomic>
#include <memory>

#include <alpaka/alpaka.hpp>

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HostOnlyTask.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The EDMetadata class provides the exact synchronization
   * mechanisms for Event data products for backends with asynchronous
   * Queue. These include
   * - adding a notification for edm::WaitingTaskWithArenaHolder
   * - recording an Event
   * - synchronizing an Event data product and a consuming EDModule
   *
   * For synchronous backends the EDMetadata acts as an owner of the
   * Queue object, as no further synchronization is needed.
   *
   * EDMetadata is used as the Metadata class for
   * edm::DeviceProduct<T>, and is an implementation detail (not
   * visible to user code).
   *
   * TODO: What to do with device-synchronous backends? The data
   * product needs to be wrapped into the edm::DeviceProduct, but the
   * EDMetadata class used there does not need anything except "dummy"
   * implementation of synchronize(). The question is clearly
   * solvable, so maybe leave it to the time we would actually need
   * one?
   */

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // Host backends with a synchronous queue

  class EDMetadata {
  public:
    EDMetadata(std::shared_ptr<Queue> queue) : queue_(std::move(queue)) {}

    Device device() const { return alpaka::getDev(*queue_); }

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const { return *queue_; }

    void recordEvent() {}

  private:
    std::shared_ptr<Queue> queue_;
  };

  // TODO: else if device backends with a synchronous queue

#else
  // All backends with an asynchronous queue

  class EDMetadata {
  public:
    EDMetadata(std::shared_ptr<Queue> queue, std::shared_ptr<Event> event)
        : queue_(std::move(queue)), event_(std::move(event)) {}
    ~EDMetadata();

    Device device() const { return alpaka::getDev(*queue_); }

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const { return *queue_; }

    void enqueueCallback(edm::WaitingTaskWithArenaHolder holder);

    void recordEvent() { alpaka::enqueue(*queue_, *event_); }

    /**
     * Synchronizes 'consumer' metadata wrt. 'this' in the event product
     */
    void synchronize(EDMetadata& consumer, bool tryReuseQueue) const;

  private:
    /**
     * Returns a shared_ptr to the Queue if it can be reused, or a
     * null shared_ptr if not
     */
    std::shared_ptr<Queue> tryReuseQueue_() const;

    std::shared_ptr<Queue> queue_;
    std::shared_ptr<Event> event_;
    // This flag tells whether the Queue may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same queue.
    mutable std::atomic<bool> mayReuseQueue_ = true;
  };
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
