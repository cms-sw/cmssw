#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitQuadrupletGeneratorForPhotonConversion.h"
#include "RecoTracker/ConversionSeedGenerators/interface/HitQuadrupletGeneratorFromLayerPairForPhotonConversion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");

  SeedingLayerSetsBuilder layerBuilder(cfg.getParameter<edm::ParameterSet>("SeedingLayers"), iC);

  SeedingLayerSets layerSets  =  layerBuilder.layers();

  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    theGenerators.emplace_back( new HitQuadrupletGeneratorFromLayerPairForPhotonConversion( set[0], set[1], &theLayerCache, 0, theMaxElement));
  }
}

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const CombinedHitQuadrupletGeneratorForPhotonConversion & cb) {
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
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end() && result.size() < maxHitQuadruplets; i++) {
    (**i).hitPairs(region, result, ev, es); 
  }
  theLayerCache.clear();
}
