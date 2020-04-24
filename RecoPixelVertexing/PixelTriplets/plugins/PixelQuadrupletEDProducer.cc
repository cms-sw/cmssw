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
#include "RecoPixelVertexing/PixelTriplets/interface/IntermediateHitTriplets.h"

#include "PixelQuadrupletGenerator.h"
#include "LayerQuadruplets.h"

class PixelQuadrupletEDProducer: public edm::stream::EDProducer<> {
public:
  PixelQuadrupletEDProducer(const edm::ParameterSet& iConfig);
  ~PixelQuadrupletEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitTriplets> tripletToken_;

  edm::RunningAverage localRA_;

  PixelQuadrupletGenerator generator_;
};

PixelQuadrupletEDProducer::PixelQuadrupletEDProducer(const edm::ParameterSet& iConfig):
  tripletToken_(consumes<IntermediateHitTriplets>(iConfig.getParameter<edm::InputTag>("triplets"))),
  generator_(iConfig, consumesCollector())
{
  produces<RegionsSeedingHitSets>();
}

void PixelQuadrupletEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("triplets", edm::InputTag("hitTripletEDProducer"));
  PixelQuadrupletGenerator::fillDescriptions(desc);

  descriptions.add("pixelQuadrupletEDProducer", desc);
}

void PixelQuadrupletEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitTriplets> htriplets;
  iEvent.getByToken(tripletToken_, htriplets);
  const auto& regionTriplets = *htriplets;

  const SeedingLayerSetsHits& seedingLayerHits = regionTriplets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < 4) {
    throw cms::Exception("LogicError") << "PixelQuadrupletEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 4, got " << seedingLayerHits.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(regionTriplets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionTriplets.regionSize(), localRA_.upper());

  // match-making of triplet and quadruplet layers
  std::vector<LayerQuadruplets::LayerSetAndLayers> quadlayers = LayerQuadruplets::layers(seedingLayerHits);

  LogDebug("PixelQuadrupletEDProducer") << "Creating quadruplets for " << regionTriplets.regionSize() << " regions, and " << quadlayers.size() << " triplet+4th layers from " << regionTriplets.tripletsSize() << " triplets";

  OrderedHitSeeds quadruplets;
  quadruplets.reserve(localRA_.upper());

  for(const auto& regionLayerTriplets: regionTriplets) {
    const TrackingRegion& region = regionLayerTriplets.region();
    auto seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    LayerHitMapCache hitCache;
    hitCache.extend(regionLayerTriplets.layerHitMapCache());

    LogTrace("PixelQuadrupletEDProducer") << " starting region, number of layer triplets " << regionLayerTriplets.layerTripletsSize();

    for(const auto& layerTriplet: regionLayerTriplets) {
      LogTrace("PixelQuadrupletEDProducer") << "  starting layer triplet " << layerTriplet.innerLayerIndex() << "," << layerTriplet.middleLayerIndex() << "," << layerTriplet.outerLayerIndex();
      auto found = std::find_if(quadlayers.begin(), quadlayers.end(), [&](const LayerQuadruplets::LayerSetAndLayers& a) {
          return a.first[0].index() == layerTriplet.innerLayerIndex() &&
                 a.first[1].index() == layerTriplet.middleLayerIndex() &&
                 a.first[2].index() == layerTriplet.outerLayerIndex();
        });
      if(found == quadlayers.end()) {
        auto exp = cms::Exception("LogicError") << "Did not find the layer triplet from vector<triplet+fourth layers>. This is a sign of some internal inconsistency\n";
        exp << "I was looking for layer triplet " << layerTriplet.innerLayerIndex() << "," << layerTriplet.middleLayerIndex() << "," << layerTriplet.outerLayerIndex()
            << ". Quadruplets have the following triplets:\n";
        for(const auto& a: quadlayers) {
          exp << " " << a.first[0].index() << "," << a.first[1].index() << "," << a.first[2].index() << ": 4th layers";
          for(const auto& b: a.second) {
            exp << " " << b.index();
          }
          exp << "\n";
        }
        throw exp;
      }
      const auto& fourthLayers = found->second;

      generator_.hitQuadruplets(region, quadruplets, iEvent, iSetup, layerTriplet.begin(), layerTriplet.end(), fourthLayers, hitCache);

#ifdef EDM_ML_DEBUG
      LogTrace("PixelQuadrupletEDProducer") << "  created " << quadruplets.size() << " quadruplets for layer triplet " << layerTriplet.innerLayerIndex() << "," << layerTriplet.middleLayerIndex() << "," << layerTriplet.outerLayerIndex() << " and 4th layers";
      for(const auto& l: fourthLayers) {
        LogTrace("PixelQuadrupletEDProducer") << "   " << l.index();
      }
#endif

      for(const auto& quad: quadruplets) {
        seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
      }
      quadruplets.clear();
    }
  }
  localRA_.update(seedingHitSets->size());

  seedingHitSets->shrink_to_fit();
  iEvent.put(std::move(seedingHitSets));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelQuadrupletEDProducer);
