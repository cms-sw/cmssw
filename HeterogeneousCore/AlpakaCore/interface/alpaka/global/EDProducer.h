#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_global_EDProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_global_EDProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace global {
    template <typename... Args>
    class EDProducer : public ProducerBase<edm::global::EDProducer, Args...> {
      static_assert(not edm::CheckAbility<edm::module::Abilities::kExternalWork, Args...>::kHasIt,
                    "ALPAKA_ACCELERATOR_NAMESPACE::global::EDProducer may not be used with ExternalWork ability. "
                    "Please use ALPAKA_ACCELERATOR_NAMESPACE::stream::SynchronizingEDProducer instead.");

    public:
      void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& iSetup) const final {
        detail::EDMetadataSentry sentry(sid);
        device::Event ev(iEvent, sentry.metadata());
        device::EventSetup const es(iSetup, ev.device());
        produce(sid, ev, es);
        this->putBackend(iEvent);
        sentry.finish();
      }

      virtual void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const = 0;
    };
  }  // namespace global
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
