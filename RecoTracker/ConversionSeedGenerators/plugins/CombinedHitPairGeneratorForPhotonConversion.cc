#include "CombinedHitPairGeneratorForPhotonConversion.h"
#include "HitPairGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/RunningAverage.h"
namespace {
  edm::RunningAverage localRA;
}


CombinedHitPairGeneratorForPhotonConversion::CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers")))
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  maxHitPairsPerTrackAndGenerator = cfg.getParameter<unsigned int>("maxHitPairsPerTrackAndGenerator");
  theGenerator.reset(new HitPairGeneratorFromLayerPairForPhotonConversion(0, 1, &theLayerCache, 0, maxHitPairsPerTrackAndGenerator));
}



const OrderedHitPairs & CombinedHitPairGeneratorForPhotonConversion::run(
									 const ConversionRegion& convRegion,
									 const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  if (thePairs.capacity()==0) thePairs.reserve(localRA.upper());
  thePairs.clear();
  hitPairs(convRegion, region, thePairs, ev, es);
  return thePairs;
}


void CombinedHitPairGeneratorForPhotonConversion::hitPairs(
							   const ConversionRegion& convRegion,
							   const TrackingRegion& region, OrderedHitPairs  & result,
							   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  assert(layers.numberOfLayersInSet() == 2);

  for(SeedingLayerSetsHits::LayerSetIndex i=0; i<layers.size(); ++i) {
    theGenerator->hitPairs( convRegion, region, result, layers[i], ev, es);
  }

}


void CombinedHitPairGeneratorForPhotonConversion::clearCache() {
    theLayerCache.clear(); 
    localRA.update(thePairs.size()); thePairs.clear(); thePairs.shrink_to_fit();
}

