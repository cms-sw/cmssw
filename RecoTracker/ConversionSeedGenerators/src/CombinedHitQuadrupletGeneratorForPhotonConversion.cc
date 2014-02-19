#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitQuadrupletGeneratorForPhotonConversion.h"
#include "RecoTracker/ConversionSeedGenerators/interface/HitQuadrupletGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers")))
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  theGenerator.reset(new HitQuadrupletGeneratorFromLayerPairForPhotonConversion( 0, 1, &theLayerCache, 0, theMaxElement));
}

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const CombinedHitQuadrupletGeneratorForPhotonConversion & cb)
  : theSeedingLayerToken(cb.theSeedingLayerToken)
{
  theMaxElement = cb.theMaxElement;
  theGenerator.reset(new HitQuadrupletGeneratorFromLayerPairForPhotonConversion( 0, 1, &theLayerCache, 0, theMaxElement));
}


CombinedHitQuadrupletGeneratorForPhotonConversion::~CombinedHitQuadrupletGeneratorForPhotonConversion() {}

void CombinedHitQuadrupletGeneratorForPhotonConversion::setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) {
  assert(0 == "not implemented");
}

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
  assert(hlayers->numberOfLayersInSet() == 2);

  for(SeedingLayerSetsHits::LayerSetIndex i=0; i<hlayers->size() && result.size() < maxHitQuadruplets; ++i) {
    theGenerator->setSeedingLayers((*hlayers)[i]);
    theGenerator->hitPairs( region, result, ev, es);
  }
  theLayerCache.clear();
}
