#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_stream_SynchronizingFixedQueueEDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_stream_SynchronizingFixedQueueEDProducer_h

#include <memory>

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::stream {

  template <typename... Args>
  class SynchronizingFixedQueueEDProducer : public ProducerBase<edm::stream::EDProducer, edm::ExternalWork, Args...> {
    static_assert(not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
                  "ExternalWork ability is redundant with "
                  "ALPAKA_ACCELERATOR_NAMESPACE::stream::SynchronizingFixedQueueEDProducer. Please remove it.");
    using Base = ProducerBase<edm::stream::EDProducer, edm::ExternalWork, Args...>;

  protected:
    SynchronizingFixedQueueEDProducer(edm::ParameterSet const iConfig) : Base(iConfig) {}

  public:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder holder) final {
      // Use the SynchronizingFixedQueueEDProducer queue for the Event metadata.
      detail::EDMetadataAcquireSentry sentry(queue_, std::move(holder), this->synchronize());
      device::Event const ev(iEvent, sentry.metadata());
      device::EventSetup const es(iSetup, ev.device());
      acquire(ev, es);
      metadata_ = sentry.finish();
    }

    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
      detail::EDMetadataSentry sentry(std::move(metadata_), this->synchronize());
      device::Event ev(iEvent, sentry.metadata());
      device::EventSetup const es(iSetup, ev.device());
      produce(ev, es);
      this->putBackend(iEvent);
      sentry.finish(ev.wasQueueUsed());
    }

    void beginStream(edm::StreamID sid) final {
      queue_ = std::make_shared<Queue>(detail::chooseDevice(sid));
      this->beginStream(sid, *queue_);
    }

    void endStream() final {
      this->endStream(*queue_);
      queue_.reset();
    }

    virtual void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) = 0;

    virtual void produce(device::Event& iEvent, device::EventSetup const& iSetup) = 0;

    // "queue" cannot be passed by const reference, because submitting any work to a queue is a non-const operation.
    // Passing it by non-const reference would be unsafe, as the user code could reset or modify the queue itself.
    // Since Queue objects have a shared_ptr semantic, it can safely be passed by value.
    virtual void beginStream(edm::StreamID sid, Queue queue) {}
    virtual void endStream(Queue queue) {}

  private:
    std::shared_ptr<Queue> queue_;
    std::shared_ptr<EDMetadata> metadata_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::stream

#endif  // HeterogeneousCore_AlpakaCore_interface_alpaka_stream_SynchronizingFixedQueueEDProducer_h
