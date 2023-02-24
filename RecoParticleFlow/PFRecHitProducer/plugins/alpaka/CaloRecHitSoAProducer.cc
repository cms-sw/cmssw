#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class CaloRecHitSoAProducer : public global::EDProducer<> {
  public:
    CaloRecHitSoAProducer(edm::ParameterSet const& config) :
      recHitsToken(consumes(config.getParameter<edm::InputTag>("src"))),
      deviceToken(produces())
    {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      const edm::SortedCollection<HBHERecHit>& recHits = event.get(recHitsToken);
      const int32_t num_recHits = recHits.size();
      if(DEBUG)
        printf("Found %d recHits\n", num_recHits);

      CaloRecHitHostCollection hostProduct{num_recHits, event.queue()};
      auto& view = hostProduct.view();

      for(int i = 0; i < num_recHits; i++)
      {
        const HBHERecHit& rh = recHits[i];
        view[i].detId() = rh.id().rawId();
        view[i].energy() = rh.energy();
        view[i].time() = rh.time();

        if (DEBUG && i < 10)
          printf("recHit %4d %u %f %f\n", i, rh.id().rawId(), rh.energy(), rh.time());
      }

      CaloRecHitDeviceCollection deviceProduct{num_recHits, event.queue()};
      alpaka::memcpy(event.queue(), deviceProduct.buffer(), hostProduct.buffer());
      event.emplace(deviceToken, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
    const device::EDPutToken<CaloRecHitDeviceCollection> deviceToken;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CaloRecHitSoAProducer);
