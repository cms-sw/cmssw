#include "CombinedHitQuadrupletGeneratorForPhotonConversion.h"
#include "HitQuadrupletGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers"))),
    theMaxElement(cfg.getParameter<unsigned int>("maxElement"))
{
  theGenerator = std::make_unique<HitQuadrupletGeneratorFromLayerPairForPhotonConversion>( 0, 1, &theLayerCache, theMaxElement);
}


CombinedHitQuadrupletGeneratorForPhotonConversion::~CombinedHitQuadrupletGeneratorForPhotonConversion() {}

const OrderedHitPairs & CombinedHitQuadrupletGeneratorForPhotonConversion::run(const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  thePairs.clear();
  hitPairs(region, thePairs, ev, es);
  return thePairs;
}


void CombinedHitQuadrupletGeneratorForPhotonConversion::hitPairs(const TrackingRegion& region, OrderedHitPairs  & result,
								 const edm::Event& ev, const edm::EventSetup& es)
{
  size_t maxHitQuadruplets=1000000;
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  assert(layers.numberOfLayersInSet() == 2);


  for(SeedingLayerSetsHits::LayerSetIndex i=0; i<hlayers->size() && result.size() < maxHitQuadruplets; ++i) {
    theGenerator->hitPairs( region, result, layers[i], ev, es);
  }
  theLayerCache.clear();
}
