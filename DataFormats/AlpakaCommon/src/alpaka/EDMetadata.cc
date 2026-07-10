#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  EDMetadata::~EDMetadata() noexcept { synchronize(); }

  void EDMetadata::synchronize() const noexcept {
    // Make sure that the production of the product in the GPU is
    // complete before destructing the product. This will also make sure
    // that the EDM stream does not move to the next event before all
    // asynchronous processing of the current is complete.

    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.

    if (eventComplete_) {
      return;
    }

    // If the event has not been set, the produce() function that
    // constructed this EDMetadata object did not launch any
    // asynchronous work.
    if (not event_) {
      return;
    }

    // This function is called from a destructor, so we want
    // to avoid throwing exceptions. If there was an
    // exception we could not really propagate it anyway.
    CMS_SA_ALLOW try { alpaka::wait(*event_); } catch (...) {
    }
    eventComplete_ = true;
  }

  // "consumer" is the metadata from an EDProducer's device::Event.
  // If the consumer queue is not set, try to reuse the queue from the
  // device product, or allocate a new one.
  void EDMetadata::synchronize(EDMetadata& consumer) const {
    assert(queue_);

    if (consumer.queue_) {
      if (*queue_ == *consumer.queue_) {
        // The consumer uses the same queue as the product, no
        // synchronisation is required.
        return;
      } else {
        // The consumer uses a different queue than the product.
        // Make sure they use the same device.
        // TODO: is this check really necessary?
        if (device_ != consumer.device_) {
          throw edm::Exception(edm::errors::LogicError) << "Handling data from multiple devices is not yet supported";
        }
      }
    } else {
      // The consumer does not have a queue, yet.
      if (tryReuseQueue_()) {
        // The consumer will use the same queue as the product, no
        // synchronisation is required.
        consumer.queue_ = queue_;
        return;
      } else {
        // The consumer will use a new queue.
        // Make sure they use the same device.
        // TODO: is this check really necessary?
        if (device_ != consumer.device_) {
          throw edm::Exception(edm::errors::LogicError) << "Handling data from multiple devices is not yet supported";
        }
        consumer.queue_ = cms::alpakatools::getQueueCache<Queue>().get(consumer.device_);
      }
    }

    if (eventComplete_) {
      return;
    }

    // If the event has not been set, the produce() function that
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

  bool EDMetadata::tryReuseQueue_() const {
    bool expected = true;
    // If the current thread is the one flipping the flag, it may
    // reuse the queue.
    return mayReuseQueue_.compare_exchange_strong(expected, false);
  }
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
