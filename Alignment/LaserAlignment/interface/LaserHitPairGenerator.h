#ifndef LaserAlignment_LaserHitPairGenerator_h
#define LaserAlignment_LaserHitPairGenerator_h

/** \class LaserHitPairGenerator
 *  generate hit pairs from hits on consecutive discs in the endcaps used by the LaserSeedGenerator
 *
 *  $Date: 2007/12/04 23:51:42 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */


#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"


#include "Alignment/LaserAlignment/interface/OrderedLaserHitPairs.h"

#include <vector>

class SeedLayerPairs;
class LayerWithHits;
class DetLayer;
class TrackingRegion;
class LaserHitPairGeneratorFromLayerPair;


class LaserHitPairGenerator
{
  typedef std::vector<LaserHitPairGeneratorFromLayerPair *> Container;
  typedef LayerHitMapCache LayerCacheType;

 public:
   /// constructor
   LaserHitPairGenerator(unsigned int size=30000) {thePairs.reserve(size); }

	/// constructor
  LaserHitPairGenerator(SeedLayerPairs & layers, const edm::EventSetup & iSetup);
	/// default constructor
  LaserHitPairGenerator(SeedLayerPairs & layers);

  /// destructor
  virtual ~LaserHitPairGenerator();

  /// add generators based on layers
  void add(const LayerWithHits * inner, const LayerWithHits * outer, const edm::EventSetup & iSetup);

  virtual void hitPairs(const TrackingRegion & reg, OrderedLaserHitPairs & prs, const edm::EventSetup & iSetup);
  virtual void hitPairs(const TrackingRegion & reg, OrderedLaserHitPairs & prs, const edm::Event & ev, const edm::EventSetup & iSetup) {}
  
  virtual const OrderedLaserHitPairs & run(const TrackingRegion& region, const edm::Event & iEvent, const edm::EventSetup& iSetup);
  
  virtual LaserHitPairGenerator * clone() const { return new LaserHitPairGenerator(*this); }

 private:
  OrderedLaserHitPairs thePairs;
  LayerCacheType theLayerCache;
  Container theGenerators;

};

#endif
