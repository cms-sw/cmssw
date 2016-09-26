#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include "CAHitQuadrupletGenerator.h"

class CAHitQuadrupletEDProducer: public edm::stream::EDProducer<> {
public:
  CAHitQuadrupletEDProducer(const edm::ParameterSet& iConfig);
  ~CAHitQuadrupletEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::RunningAverage localRA_;

  CAHitQuadrupletGenerator generator_;
};

CAHitQuadrupletEDProducer::CAHitQuadrupletEDProducer(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
  generator_(iConfig, consumesCollector(), false)
{
  produces<RegionsSeedingHitSets>();
}

void CAHitQuadrupletEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  CAHitQuadrupletGenerator::fillDescriptions(desc);

  descriptions.add("caHitQuadrupletEDProducer", desc);
}

void CAHitQuadrupletEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < 4) {
    throw cms::Exception("Configuration") << "CAHitQuadrupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 4, got " << seedingLayerHits.numberOfLayersInSet();
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  generator_.initEvent(iEvent, iSetup);

  LogDebug("CAHitQuadrupletEDProducer") << "Creating quadruplets for " << regionDoublets.regionSize() << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";

  OrderedHitSeeds quadruplets;
  quadruplets.reserve(localRA_.upper());

  for(const auto& regionLayerPairs: regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    LogTrace("CAHitQuadrupletEDProducer") << " starting region";

    generator_.hitQuadruplets(regionLayerPairs, quadruplets, iSetup, seedingLayerHits);
    LogTrace("CAHitQuadrupletEDProducer") << "  created " << quadruplets.size() << " quadruplets";

    for(const auto& quad: quadruplets) {
      seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
    }
    quadruplets.clear();
  }
  localRA_.update(seedingHitSets->size());

  iEvent.put(std::move(seedingHitSets));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CAHitQuadrupletEDProducer);
