#ifndef PixelHitTripletGenerator_H
#define PixelHitTripletGenerator_H

/** A HitTripletGenerator consisting of a set of 
 *  triplet generators of type HitTripletGeneratorFromPairAndLayers
 *  initialised from provided layers in the form of PixelLayerTriplets  
 */ 


class TrackingRegion;
class LayerWithHits;
class DetLayer;
class HitTripletGeneratorFromPairAndLayers;
class PixelLayerTriplets;
//class SiPixelRecHitCollection;

namespace edm { class Event; }
namespace edm { class EventSetup; }

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>


class PixelHitTripletGenerator : public HitTripletGenerator {

  typedef std::vector<HitTripletGeneratorFromPairAndLayers* > GeneratorContainer;

public:
  PixelHitTripletGenerator( const edm::ParameterSet& cfg);
  virtual ~PixelHitTripletGenerator();

  typedef LayerHitMapCache  LayerCacheType;


  /// from base class
  virtual void hitTriplets( const TrackingRegion&, OrderedHitTriplets&, 
     const edm::EventSetup& );

  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);
  void init(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
private:

  edm::ParameterSet         theConfig;
  LayerCacheType            theLayerCache;
  GeneratorContainer        theGenerators;
  PixelLayerTriplets *      thePixel;

};
#endif
