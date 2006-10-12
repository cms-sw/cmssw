#ifndef PixelHitTripletGenerator_H
#define PixelHitTripletGenerator_H

/** A HitTripletGenerator consisting of a set of 
 *  triplet generators of type HitTripletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"

class TrackingRegion;
class LayerWithHits;
class DetLayer;
class HitTripletGeneratorFromPairAndLayers;
class PixelLayerTriplets;

class PixelHitTripletGenerator : public HitTripletGenerator {

  typedef vector<HitTripletGeneratorFromPairAndLayers* > GeneratorContainer;

public:
  PixelHitTripletGenerator();

  typedef LayerHitMapCache  LayerCacheType;

  virtual ~PixelHitTripletGenerator();

  /// from base class
  virtual void hitTriplets( const TrackingRegion&, OrderedHitTriplets&, 
     const edm::EventSetup& );

  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);
  
private:

  LayerCacheType            theLayerCache;
  GeneratorContainer        theGenerators;
  PixelLayerTriplets * thePixel;

};
#endif
