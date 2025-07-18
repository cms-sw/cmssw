#include <alpaka/alpaka.hpp>

#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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

    // If event_ is null, the EDMetadata was either
    // default-constructed, or fully synchronized before leaving the
    // produce() call, so no synchronization is needed.
    // If the queue was re-used, then some other EDMetadata object in
    // the same edm::Event records the event_ (in the same queue) and
    // calls alpaka::wait(), and therefore this wait() call can be
    // skipped).
    if (event_ and not eventComplete_ and mayReuseQueue_) {
      // Must not throw in a destructor, and if there were an
      // exception could not really propagate it anyway.
      CMS_SA_ALLOW try { alpaka::wait(*event_); } catch (...) {
      }
    }
  }

  void EDMetadata::enqueueCallback(edm::WaitingTaskWithArenaHolder holder) {
    edm::Service<edm::Async> async;
    recordEvent();
    async->runAsync(
        std::move(holder),
        [event = event_]() mutable { alpaka::wait(*event); },
        []() { return "Enqueued via " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "::EDMetadata::enqueueCallback()"; });
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

    if (eventComplete_) {
      return;
    }

    // TODO: how necessary this check is?
    if (alpaka::getDev(*queue_) != alpaka::getDev(*consumer.queue_)) {
      throw edm::Exception(edm::errors::LogicError) << "Handling data from multiple devices is not yet supported";
    }

    // If the event has been discarded, the produce() function that
    // constructed this EDMetadata object did not launch any
    // asynchronous work.
    if (not event_) {
      return;
    }

    if (alpaka::isComplete(*event_)) {
      eventComplete_ = true;
    } else {
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
