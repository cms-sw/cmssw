#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

/**
 * This class will eventually be the one creating the reco::Track
 * objects from the output of GPU CA. Now it is just to produce
 * something persistable.
 */
class PixelTrackProducerFromCUDA: public edm::global::EDProducer<> {
 public:
  explicit PixelTrackProducerFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackProducerFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

 private:
  edm::EDGetTokenT<RegionsSeedingHitSets> srcToken_;
  bool enabled_;
};

PixelTrackProducerFromCUDA::PixelTrackProducerFromCUDA(const edm::ParameterSet& iConfig):
  srcToken_(consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<int>();
}

void PixelTrackProducerFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));
  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromCUDA::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.put(std::make_unique<int>(0));
}

DEFINE_FWK_MODULE(PixelTrackProducerFromCUDA);
