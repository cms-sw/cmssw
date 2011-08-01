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

CombinedHitPairGeneratorForPhotonConversion::CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  maxHitPairsPerTrackAndGenerator = cfg.getParameter<unsigned int>("maxHitPairsPerTrackAndGenerator");

}

void CombinedHitPairGeneratorForPhotonConversion::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
  theMaxElement = cfg.getParameter<unsigned int>("maxElement");
  maxHitPairsPerTrackAndGenerator = cfg.getParameter<unsigned int>("maxHitPairsPerTrackAndGenerator");

  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers(es); 
  init(layerSets);
}

void CombinedHitPairGeneratorForPhotonConversion::init(const SeedingLayerSets & layerSets)
{
  initialised = true;
  typedef SeedingLayerSets::const_iterator IL;
  for (IL il=layerSets.begin(), ilEnd=layerSets.end(); il != ilEnd; ++il) {
    const SeedingLayers & set = *il;
    if (set.size() != 2) continue;
    add( set[0], set[1] );
  }
}

void CombinedHitPairGeneratorForPhotonConversion::cleanup()
{
  Container::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
  theGenerators.clear();
}

CombinedHitPairGeneratorForPhotonConversion::~CombinedHitPairGeneratorForPhotonConversion() { cleanup(); }

void CombinedHitPairGeneratorForPhotonConversion::add( const SeedingLayer& inner, const SeedingLayer& outer)
{ 
  theGenerators.push_back( new HitPairGeneratorFromLayerPairForPhotonConversion( inner, outer, &theLayerCache, 0, maxHitPairsPerTrackAndGenerator));
}

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
  if (theESWatcher.check(es) || !initialised ) {
    cleanup();
    init(theConfig,es);
  }

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
