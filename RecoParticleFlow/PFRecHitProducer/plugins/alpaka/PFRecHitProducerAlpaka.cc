#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitProducerKernel.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFRecHitProducerAlpaka : public global::EDProducer<> {
  public:
    PFRecHitProducerAlpaka(edm::ParameterSet const& config)
        : recHitsToken(consumes(config.getParameter<edm::InputTag>("src"))), pfRecHitsToken(produces()) {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      const CaloRecHitDeviceCollection& recHits = event.get(recHitsToken);
      const int num_recHits = recHits->metadata().size();
      PFRecHitDeviceCollection pfRecHits{num_recHits, event.queue()};

      PFRecHitProducerKernel kernel{};
      kernel.execute(event.queue(), recHits, pfRecHits);

      event.emplace(pfRecHitsToken, std::move(pfRecHits));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<CaloRecHitDeviceCollection> recHitsToken;
    const device::EDPutToken<PFRecHitDeviceCollection> pfRecHitsToken;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFRecHitProducerAlpaka);
