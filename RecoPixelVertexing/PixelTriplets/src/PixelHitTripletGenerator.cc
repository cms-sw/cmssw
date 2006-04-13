#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelLayerTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

void PixelHitTripletGenerator::
    hitTriplets( const TrackingRegion& region, OrderedHitTriplets & result, const edm::EventSetup& iSetup) 
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
}

void PixelHitTripletGenerator::init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup)
{
  PixelLayerTriplets pixel;
  pixel.init(coll, iSetup);
  vector<PixelLayerTriplets::LayerPairAndLayers>::const_iterator it;
  vector<PixelLayerTriplets::LayerPairAndLayers> trilayers=pixel.layers();
  for (it = trilayers.begin(); it != trilayers.end(); it++) {
    const LayerWithHits * first = (*it).first.first;
    const LayerWithHits * second = (*it).first.second;
    vector<const LayerWithHits *> thirds = (*it).second;

    theGenerators.push_back( new HitTripletGeneratorFromPairAndLayers(
        HitPairGeneratorFromLayerPair( first, second, &theLayerCache, iSetup),
        thirds, 
        &theLayerCache) );
  } 
}
  
