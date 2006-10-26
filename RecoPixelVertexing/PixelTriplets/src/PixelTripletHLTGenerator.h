#ifndef PixelTripletHLTGenerator_H
#define PixelTripletHLTGenerator_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"


class PixelTripletHLTGenerator : public HitTripletGeneratorFromPairAndLayers {

typedef PixelHitTripletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelTripletHLTGenerator( const edm::ParameterSet& cfg) 
    : theConfig(cfg), thePairGenerator(0), theLayerCache(0) { }

  virtual ~PixelTripletHLTGenerator() { delete thePairGenerator; }

  virtual void init( const HitPairGenerator & pairs,
      std::vector<const LayerWithHits*> layers, LayerCacheType* layerCache);

  virtual void hitTriplets(
      const TrackingRegion& region, OrderedHitTriplets & trs, const edm::EventSetup& iSetup);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
  const vector<const LayerWithHits*> thirdLayers() const { return theLayers; }

private:

  edm::ParameterSet         theConfig;
  HitPairGenerator * thePairGenerator;
  vector<const LayerWithHits*> theLayers;
  LayerCacheType * theLayerCache;
};
#endif


