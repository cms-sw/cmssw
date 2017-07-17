#include "CombinedHitQuadrupletGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "LayerQuadruplets.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


using namespace std;
using namespace ctfseeding;

CombinedHitQuadrupletGenerator::CombinedHitQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers")))
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  edm::ParameterSet tripletGeneratorPSet = cfg.getParameter<edm::ParameterSet>("TripletGeneratorPSet");
  std::string tripletGeneratorName = tripletGeneratorPSet.getParameter<std::string>("ComponentName");

  std::unique_ptr<HitTripletGeneratorFromPairAndLayers> tripletGenerator(HitTripletGeneratorFromPairAndLayersFactory::get()->create(tripletGeneratorName, tripletGeneratorPSet, iC));
  // Some CPU wasted here because same pairs are generated multiple times
  tripletGenerator->init(std::make_unique<HitPairGeneratorFromLayerPair>(0, 1, &theLayerCache), &theLayerCache);

  theGenerator.reset(HitQuadrupletGeneratorFromTripletAndLayersFactory::get()->create(generatorName, generatorPSet, iC));
  theGenerator->init(std::move(tripletGenerator), &theLayerCache);
}

CombinedHitQuadrupletGenerator::~CombinedHitQuadrupletGenerator() {}

void CombinedHitQuadrupletGenerator::hitQuadruplets(
   const TrackingRegion& region, OrderedHitSeeds & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  if(layers.numberOfLayersInSet() != 4)
    throw cms::Exception("Configuration") << "CombinedHitQuadrupletsGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 4, got " << layers.numberOfLayersInSet();

  std::vector<LayerQuadruplets::LayerSetAndLayers> quadlayers = LayerQuadruplets::layers(layers);
  for(const auto& tripletAndLayers: quadlayers) {
    theGenerator->hitQuadruplets(region, result, ev, es, tripletAndLayers.first, tripletAndLayers.second);
  }
  theLayerCache.clear();
}
