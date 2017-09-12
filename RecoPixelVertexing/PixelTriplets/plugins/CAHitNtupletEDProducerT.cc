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

namespace {
  void fillNtuplets(RegionsSeedingHitSets::RegionFiller& seedingHitSetsFiller,
                    const OrderedHitSeeds& quadruplets) {
    for(const auto& quad: quadruplets) {
      seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
    }
  }
}

template <typename T_Generator>
class CAHitNtupletEDProducerT: public edm::stream::EDProducer<> {
public:
  CAHitNtupletEDProducerT(const edm::ParameterSet& iConfig);
  ~CAHitNtupletEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::RunningAverage localRA_;

  T_Generator generator_;
};

template <typename T_Generator>
CAHitNtupletEDProducerT<T_Generator>::CAHitNtupletEDProducerT(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
  generator_(iConfig, consumesCollector())
{
  produces<RegionsSeedingHitSets>();
}

template <typename T_Generator>
void CAHitNtupletEDProducerT<T_Generator>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  T_Generator::fillDescriptions(desc);

  auto label = T_Generator::fillDescriptionsLabel() + std::string("EDProducer");
  descriptions.add(label, desc);
}

template <typename T_Generator>
void CAHitNtupletEDProducerT<T_Generator>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < T_Generator::minLayers) {
    throw cms::Exception("LogicError") << "CAHitNtupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= " << T_Generator::minLayers << ", got " << seedingLayerHits.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  generator_.initEvent(iEvent, iSetup);

  LogDebug("CAHitNtupletEDProducer") << "Creating ntuplets for " << regionDoublets.regionSize() << " regions, and " << regionDoublets.layerPairsSize() << " layer pairs";
  std::vector<OrderedHitSeeds> ntuplets;
  ntuplets.resize(regionDoublets.regionSize());
  for(auto& ntuplet : ntuplets)  ntuplet.reserve(localRA_.upper()); 
   
  generator_.hitNtuplets(regionDoublets, ntuplets, iSetup, seedingLayerHits);
  int index = 0;
  for(const auto& regionLayerPairs: regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    fillNtuplets(seedingHitSetsFiller, ntuplets[index]);
    ntuplets[index].clear();
    index++;
  }
  localRA_.update(seedingHitSets->size());

  iEvent.put(std::move(seedingHitSets));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CAHitQuadrupletGenerator.h"
using CAHitQuadrupletEDProducer = CAHitNtupletEDProducerT<CAHitQuadrupletGenerator>;
DEFINE_FWK_MODULE(CAHitQuadrupletEDProducer);

#include "CAHitTripletGenerator.h"
using CAHitTripletEDProducer = CAHitNtupletEDProducerT<CAHitTripletGenerator>;
DEFINE_FWK_MODULE(CAHitTripletEDProducer);
