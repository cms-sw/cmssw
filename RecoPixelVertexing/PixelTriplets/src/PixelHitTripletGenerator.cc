#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelLayerTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletHLTGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

PixelHitTripletGenerator::PixelHitTripletGenerator(const edm::ParameterSet& cfg )
 : theConfig(cfg), thePixel(0) {}

void PixelHitTripletGenerator::
    hitTriplets( const TrackingRegion& region, OrderedHitTriplets & result, 
        const edm::EventSetup& iSetup) 
{

  GeneratorContainer::const_iterator ic;
  for (ic = theGenerators.begin(); ic != theGenerators.end(); ic++) {
    (**ic).hitTriplets( region, result, iSetup);
  }
  theLayerCache.clear();
}

PixelHitTripletGenerator::~PixelHitTripletGenerator()
{
  GeneratorContainer::const_iterator ig;
  for (ig = theGenerators.begin(); ig != theGenerators.end(); ig++) {
    delete (*ig);
  }
  delete thePixel;
}

void PixelHitTripletGenerator::init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup)
{
  if (!thePixel) thePixel = new PixelLayerTriplets;
  thePixel->init(coll, iSetup);
  vector<PixelLayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<PixelLayerTriplets::LayerPairAndLayers> trilayers=thePixel->layers();
  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    const LayerWithHits * first = (*it).first.first;
    const LayerWithHits * second = (*it).first.second;
    vector<const LayerWithHits *> thirds = (*it).second;

     
    std::string       generatorName = theConfig.getParameter<std::string>("Generator");
    edm::ParameterSet generatorPSet = theConfig.getParameter<edm::ParameterSet>("GeneratorPSet");

    HitTripletGeneratorFromPairAndLayers * aGen = 
 //       HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName,generatorPSet);
    new PixelTripletHLTGenerator(generatorPSet);

    aGen->init( HitPairGeneratorFromLayerPair( first, second, &theLayerCache, iSetup),
                thirds, &theLayerCache); 

    theGenerators.push_back( aGen);
  } 
}
  
