#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_stream_EDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_stream_EDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace stream {
    template <typename... Args>
    class EDProducer : public ProducerBase<edm::stream::EDProducer, Args...> {
      static_assert(not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
                    "ALPAKA_ACCELERATOR_NAMESPACE::stream::EDProducer may not be used with ExternalWork ability. "
                    "Please use ALPAKA_ACCELERATOR_NAMESPACE::stream::SynchronizingEDProducer instead.");
      using Base = ProducerBase<edm::stream::EDProducer, Args...>;

    protected:
      EDProducer(edm::ParameterSet const iConfig) : Base(iConfig) {}

    public:
      void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
        detail::EDMetadataSentry sentry(iEvent.streamID(), this->synchronize());
        device::Event ev(iEvent, sentry.metadata());
        device::EventSetup const es(iSetup, ev.device());
        produce(ev, es);
        this->putBackend(iEvent);
        sentry.finish(ev.wasQueueUsed());
      }

      virtual void produce(device::Event& iEvent, device::EventSetup const& iSetup) = 0;
    };
  }  // namespace stream
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
