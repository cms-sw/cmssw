#ifndef LaserAlignment_LaserHitPairGenerator_h
#define LaserAlignment_LaserHitPairGenerator_h

/** \class LaserHitPairGenerator
 *  generate hit pairs from hits on consecutive discs in the endcaps used by the LaserSeedGenerator
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

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
	/// constructor
  LaserHitPairGenerator(SeedLayerPairs & layers, const edm::EventSetup & iSetup);
	/// default constructor
  LaserHitPairGenerator(SeedLayerPairs & layers);

  /// destructor
  ~LaserHitPairGenerator();

  /// add generators based on layers
  void add(const LayerWithHits * inner, const LayerWithHits * outer, const edm::EventSetup & iSetup);

  /// from base class
  virtual void hitPairs(const TrackingRegion & reg, OrderedHitPairs & prs, const edm::EventSetup & iSetup);
  virtual void hitPairs(const TrackingRegion & reg, OrderedHitPairs & prs, const edm::Event & ev, const edm::EventSetup & iSetup) {}
  virtual LaserHitPairGenerator * clone() const { return new LaserHitPairGenerator(*this); }

 private:
  LayerCacheType theLayerCache;
  Container theGenerators;

};

#endif
