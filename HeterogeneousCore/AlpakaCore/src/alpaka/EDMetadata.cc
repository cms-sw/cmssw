#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  EDMetadata::~EDMetadata() {
    // Make sure that the production of the product in the GPU is
    // complete before destructing the product. This is to make sure
    // that the EDM stream does not move to the next event before all
    // asynchronous processing of the current is complete.

    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.

    if (event_) {
      // Must not throw in a destructor, and if there were an
      // exception could not really propagate it anyway.
      CMS_SA_ALLOW try { alpaka::wait(*event_); } catch (...) {
      }
    }
  }

  void EDMetadata::enqueueCallback(edm::WaitingTaskWithArenaHolder holder) {
    alpaka::enqueue(*queue_, alpaka::HostOnlyTask([holder = std::move(holder)]() {
      // The functor is required to be const, but the original waitingTaskHolder_
      // needs to be notified...
      const_cast<edm::WaitingTaskWithArenaHolder&>(holder).doneWaiting(nullptr);
    }));
  }

  void EDMetadata::synchronize(EDMetadata& consumer, bool tryReuseQueue) const {
    if (*queue_ == *consumer.queue_) {
      return;
    }

    if (tryReuseQueue) {
      if (auto queue = tryReuseQueue_()) {
        consumer.queue_ = queue_;
        return;
      }
    }

    // TODO: how necessary this check is?
    if (alpaka::getDev(*queue_) != alpaka::getDev(*consumer.queue_)) {
      throw edm::Exception(edm::errors::LogicError) << "Handling data from multiple devices is not yet supported";
    }

    if (not alpaka::isComplete(*event_)) {
      // Event not yet occurred, so need to add synchronization
      // here. Sychronization is done by making the queue to wait
      // for an event, so all subsequent work in the queue will run
      // only after the event has "occurred" (i.e. data product
      // became available).
      alpaka::wait(*consumer.queue_, *event_);
    }
  }

  std::shared_ptr<Queue> EDMetadata::tryReuseQueue_() const {
    bool expected = true;
    if (mayReuseQueue_.compare_exchange_strong(expected, false)) {
      // If the current thread is the one flipping the flag, it may
      // reuse the queue.
      return queue_;
    }
    return nullptr;
  }
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
