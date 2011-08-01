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

CombinedHitQuadrupletGeneratorForPhotonConversion::CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
}

void CombinedHitQuadrupletGeneratorForPhotonConversion::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");

  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers(es); 
  init(layerSets);
}

void CombinedHitQuadrupletGeneratorForPhotonConversion::init(const SeedingLayerSets & layerSets)
{
  initialised = true;
  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    add( set[0], set[1] );
  }
}

void CombinedHitQuadrupletGeneratorForPhotonConversion::cleanup()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
  theGenerators.clear();
}

CombinedHitQuadrupletGeneratorForPhotonConversion::~CombinedHitQuadrupletGeneratorForPhotonConversion() { cleanup(); }

void CombinedHitQuadrupletGeneratorForPhotonConversion::add( const SeedingLayer& inner, const SeedingLayer& outer)
{ 
  theGenerators.push_back( new HitQuadrupletGeneratorFromLayerPairForPhotonConversion( inner, outer, &theLayerCache, 0, theMaxElement));
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
  if (theESWatcher.check(es) || !initialised ) {
    cleanup();
    init(theConfig,es);
  }

  size_t maxHitQuadruplets=1000000;
  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end() && result.size() < maxHitQuadruplets; i++) {
    (**i).hitPairs(region, result, ev, es); 
  }
  theLayerCache.clear();
}
