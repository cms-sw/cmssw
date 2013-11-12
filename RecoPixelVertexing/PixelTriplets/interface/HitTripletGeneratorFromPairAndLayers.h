#ifndef HitTripletGeneratorFromPairAndLayers_H
#define HitTripletGeneratorFromPairAndLayers_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include <vector>
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"

namespace edm { class ConsumesCollector; }

class HitTripletGeneratorFromPairAndLayers : public HitTripletGenerator {

public:
  typedef LayerHitMapCache  LayerCacheType;

  virtual ~HitTripletGeneratorFromPairAndLayers() {}

  virtual void init( const HitPairGenerator & pairs, LayerCacheType* layerCache) = 0;

  virtual void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                                std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) = 0;
};
#endif


