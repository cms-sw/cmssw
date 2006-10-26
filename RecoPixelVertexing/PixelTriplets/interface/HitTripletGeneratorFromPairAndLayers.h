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
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"

class HitTripletGeneratorFromPairAndLayers : public HitTripletGenerator {

public:
  typedef PixelHitTripletGenerator::LayerCacheType       LayerCacheType;
  virtual ~HitTripletGeneratorFromPairAndLayers() {}
  virtual void init( const HitPairGenerator & pairs, 
    std::vector<const LayerWithHits*> layers, LayerCacheType* layerCache) = 0; 
};
#endif


