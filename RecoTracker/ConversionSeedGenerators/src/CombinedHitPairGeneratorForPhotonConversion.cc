#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitPairGeneratorForPhotonConversion.h"
#include "RecoTracker/ConversionSeedGenerators/interface/HitPairGeneratorFromLayerPairForPhotonConversion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitPairGeneratorForPhotonConversion::CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  maxHitPairsPerTrackAndGenerator = cfg.getParameter<unsigned int>("maxHitPairsPerTrackAndGenerator");

  SeedingLayerSetsBuilder layerBuilder(cfg.getParameter<edm::ParameterSet>("SeedingLayers"), iC);

  SeedingLayerSets layerSets  =  layerBuilder.layers();

  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    theGenerators.emplace_back( new HitPairGeneratorFromLayerPairForPhotonConversion( set[0], set[1], &theLayerCache, 0, maxHitPairsPerTrackAndGenerator));
  }
}

CombinedHitPairGeneratorForPhotonConversion::CombinedHitPairGeneratorForPhotonConversion(const CombinedHitPairGeneratorForPhotonConversion & cb):
  maxHitPairsPerTrackAndGenerator(cb.maxHitPairsPerTrackAndGenerator) {
  theGenerators.reserve(cb.theGenerators.size());
  for(const auto& gen: cb.theGenerators) {
    theGenerators.emplace_back(gen->clone());
  }
}


CombinedHitPairGeneratorForPhotonConversion::~CombinedHitPairGeneratorForPhotonConversion() {}


const OrderedHitPairs & CombinedHitPairGeneratorForPhotonConversion::run(
									 const ConversionRegion& convRegion,
									 const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es)
{
  thePairs.clear();
  hitPairs(convRegion, region, thePairs, ev, es);
  return thePairs;
}


void CombinedHitPairGeneratorForPhotonConversion::hitPairs(
							   const ConversionRegion& convRegion,
							   const TrackingRegion& region, OrderedHitPairs  & result,
							   const edm::Event& ev, const edm::EventSetup& es)
{
  Container::const_iterator i;
  OrderedHitPairs  resultTmp;
  resultTmp.reserve(maxHitPairsPerTrackAndGenerator);

  for (i=theGenerators.begin(); i!=theGenerators.end() && result.size() < theMaxElement; i++) {
    resultTmp.clear();
    (**i).hitPairs(convRegion, region, resultTmp, ev, es); 
    result.insert(result.end(),resultTmp.begin(),resultTmp.end());
  }
  //theLayerCache.clear(); //Don't want to clear now, because have to loop on all the tracks. will be cleared later, calling a specific method
}
