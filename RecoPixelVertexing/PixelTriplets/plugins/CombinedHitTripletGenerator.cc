#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "LayerTriplets.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace ctfseeding;

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg)
  : initialised(false), theConfig(cfg)
{ }

void CombinedHitTripletGenerator::init(const edm::ParameterSet & cfg, const edm::EventSetup& es)
{
//  edm::ParameterSet leyerPSet = cfg.getParameter<edm::ParameterSet>("LayerPSet");
//  SeedingLayerSets layerSets  = SeedingLayerSetsBuilder(leyerPSet).layers(es);

  std::string layerBuilderName = cfg.getParameter<std::string>("SeedingLayers");
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get(layerBuilderName, layerBuilder);

  SeedingLayerSets layerSets  =  layerBuilder->layers(es);


  vector<LayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<LayerTriplets::LayerPairAndLayers> trilayers=LayerTriplets(layerSets).layers();

  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    SeedingLayer first = (*it).first.first;
    SeedingLayer second = (*it).first.second;
    vector<SeedingLayer> thirds = (*it).second;

    edm::ParameterSet generatorPSet = theConfig.getParameter<edm::ParameterSet>("GeneratorPSet");
    std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");

    HitTripletGeneratorFromPairAndLayers * aGen =
        HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName,generatorPSet);

    aGen->init( HitPairGeneratorFromLayerPair( first, second, &theLayerCache),
                thirds, &theLayerCache);

    theGenerators.push_back( aGen);
  }

  initialised = true;

}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator()
{
  GeneratorContainer::const_iterator it;
  for (it = theGenerators.begin(); it!= theGenerators.end(); it++) {
    delete (*it);
  }
}


void CombinedHitTripletGenerator::hitTriplets(
   const TrackingRegion& region, OrderedHitTriplets & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  if (!initialised) init(theConfig,es);

  GeneratorContainer::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitTriplets( region, result, ev, es);
  }
  theLayerCache.clear();
}

