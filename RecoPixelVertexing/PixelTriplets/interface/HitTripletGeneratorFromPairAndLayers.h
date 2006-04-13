#ifndef HitTripletGeneratorFromPairAndLayers_H
#define HitTripletGeneratorFromPairAndLayers_H

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


class HitTripletGeneratorFromPairAndLayers : public HitTripletGenerator {

typedef PixelHitTripletGenerator::LayerCacheType       LayerCacheType;

public:
  HitTripletGeneratorFromPairAndLayers(
      const HitPairGenerator & generator,
      vector<const LayerWithHits*> layers,
      LayerCacheType* layerCache)
    : thePairGenerator(generator.clone()),
      theLayers(layers),
      theLayerCache(*layerCache)
    { }

  ~HitTripletGeneratorFromPairAndLayers() { delete thePairGenerator; }

  virtual void hitTriplets(
      const TrackingRegion& region, OrderedHitTriplets & trs, const edm::EventSetup& iSetup);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
  const vector<const LayerWithHits*> thirdLayers() const { return theLayers; }

private:
  HitPairGenerator * thePairGenerator;
  vector<const LayerWithHits*> theLayers;
  LayerCacheType & theLayerCache;
};
#endif


