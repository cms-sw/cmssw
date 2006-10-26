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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class LayerWithHits;
class DetLayer;
class HitTripletGeneratorFromPairAndLayers;
class PixelLayerTriplets;


class PixelHitTripletGenerator : public HitTripletGenerator {

  typedef vector<HitTripletGeneratorFromPairAndLayers* > GeneratorContainer;

public:
  PixelHitTripletGenerator( const edm::ParameterSet& cfg);
  virtual ~PixelHitTripletGenerator();

  typedef LayerHitMapCache  LayerCacheType;


  /// from base class
  virtual void hitTriplets( const TrackingRegion&, OrderedHitTriplets&, 
     const edm::EventSetup& );

  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);
  
private:

  edm::ParameterSet         theConfig;
  LayerCacheType            theLayerCache;
  GeneratorContainer        theGenerators;
  PixelLayerTriplets *      thePixel;

};
#endif
