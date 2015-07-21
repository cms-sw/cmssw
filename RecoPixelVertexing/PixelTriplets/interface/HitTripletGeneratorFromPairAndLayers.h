#ifndef HitTripletGeneratorFromPairAndLayers_H
#define HitTripletGeneratorFromPairAndLayers_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include <vector>
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"

namespace edm { class ParameterSet; class Event; class EventSetup; class ConsumesCollector; }
class TrackingRegion;
class HitPairGeneratorFromLayerPair;

class HitTripletGeneratorFromPairAndLayers {

public:
  typedef LayerHitMapCache  LayerCacheType;

  explicit HitTripletGeneratorFromPairAndLayers(unsigned int maxElement=0);
  explicit HitTripletGeneratorFromPairAndLayers(const edm::ParameterSet& pset);
  virtual ~HitTripletGeneratorFromPairAndLayers();

  void init( std::unique_ptr<HitPairGeneratorFromLayerPair>&& pairs, LayerCacheType* layerCache);

  const HitPairGeneratorFromLayerPair& pairGenerator() const { return *thePairGenerator; }

  virtual void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
                            const edm::Event & ev, const edm::EventSetup& es,
                            SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                            const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) = 0;
protected:
  std::unique_ptr<HitPairGeneratorFromLayerPair> thePairGenerator;
  LayerCacheType *theLayerCache;
  const unsigned int theMaxElement;
};
#endif


