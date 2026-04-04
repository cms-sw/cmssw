#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_stream_FixedQueueEDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_stream_FixedQueueEDProducer_h

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace stream {
    template <typename... Args>
    class FixedQueueEDProducer : public ProducerBase<edm::stream::EDProducer, Args...> {
      static_assert(not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
                    "ALPAKA_ACCELERATOR_NAMESPACE::stream::FixedQueueEDProducer may not be used with ExternalWork "
                    "ability. Please request this functionality to the Heterogeneous or Core Software developers.");
      using Base = ProducerBase<edm::stream::EDProducer, Args...>;

    protected:
      FixedQueueEDProducer(edm::ParameterSet const iConfig) : Base(iConfig) {}

    public:
      void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
        detail::EDMetadataSentry sentry(queue_, this->synchronize());
        // Force the device::Event to use the queue from the metadata.
        device::Event ev(iEvent, sentry.metadata(), true);
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

      virtual void produce(device::Event& iEvent, device::EventSetup const& iSetup) = 0;

      // "queue" cannot be passed by const reference, because submitting any work to a queue is a non-const operation.
      // Passing it by non-const reference would be unsafe, as the user code could reset or modify the queue itself.
      // Since Queue objects have a shared_ptr semantic, it can safely be passed by value.
      virtual void beginStream(edm::StreamID sid, Queue queue) {}
      virtual void endStream(Queue queue) {}

    private:
      std::shared_ptr<Queue> queue_;
    };
  }  // namespace stream
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaCore_interface_alpaka_stream_FixedQueueEDProducer_h
