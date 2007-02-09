/* 
 * generate hit pairs from hits on consecutive discs in the endcaps
 * used by the LaserSeedGenerator
 */

#ifndef LaserAlignment_LaserHitPairGenerator_h
#define LaserAlignment_LaserHitPairGenerator_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"

#include "DataFormats/Common/interface/RangeMap.h"

#include "Alignment/LaserAlignment/interface/LaserHitPairGeneratorFromLayerPair.h"

#include <vector>

class SeedLayerPairs;
class LayerWithHits;
class DetLayer;
class TrackingRegion;
class HitPairGeneratorFromLayerPair;


class LaserHitPairGenerator : public HitPairGenerator
{
  typedef std::vector<LaserHitPairGeneratorFromLayerPair *> Container;
  typedef LayerHitMapCache LayerCacheType;

 public:
  LaserHitPairGenerator(SeedLayerPairs & layers, const edm::EventSetup & iSetup);
  LaserHitPairGenerator(SeedLayerPairs & layers);

  ~LaserHitPairGenerator();

  // add generators based on layers
  void add(const LayerWithHits * inner, const LayerWithHits * outer, const edm::EventSetup & iSetup);

  // from base class
  virtual void hitPairs(const TrackingRegion & reg, OrderedHitPairs & prs, const edm::EventSetup & iSetup);
  virtual LaserHitPairGenerator * clone() const { return new LaserHitPairGenerator(*this); }

 private:
  LayerCacheType theLayerCache;
  Container theGenerators;

};

#endif
