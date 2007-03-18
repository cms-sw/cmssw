#ifndef LaserAlignment_LaserHitPairGeneratorFromLayerPair_h
#define LaserAlignment_LaserHitPairGeneratorFromLayerPair_h

/** \class CompareHitPairsZ
 *  compare hit pairs in z direction
 *
 *  $Date: Fri Mar 16 16:06:30 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

/** \class LaserHitPairGeneratorFromLayerPair
 *  generate hit pairs from hits on consecutive discs in the endcaps used by the LaserSeedGenerator
 *
 *  $Date: Fri Mar 16 16:05:32 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"

class DetLayer;
class TrackingRegion;
class LayerWithHits;

class CompareHitPairsZ
{
 public:
	/// compare hit pairs in z direction
  CompareHitPairsZ(const edm::EventSetup& iSetup)
     { iSetup.get<TrackerDigiGeometryRecord>().get(tracker); }

  bool operator() ( OrderedHitPair h1, OrderedHitPair h2)
  {
    GlobalPoint in1p = tracker->idToDet(h1.inner()->geographicalId())->surface().toGlobal(h1.inner()->localPosition());
    GlobalPoint in2p = tracker->idToDet(h2.inner()->geographicalId())->surface().toGlobal(h2.inner()->localPosition());
    GlobalPoint ou1p = tracker->idToDet(h1.outer()->geographicalId())->surface().toGlobal(h1.outer()->localPosition());
    GlobalPoint ou2p = tracker->idToDet(h2.outer()->geographicalId())->surface().toGlobal(h2.outer()->localPosition());

    if (ou1p.z() * ou2p.z() < 0.0) return ou1p.z() > ou2p.z();
    else
      {
	double dist1 = 100.0 * fabs(ou1p.z() - in1p.z());
	double dist2 = 100.0 * fabs(ou2p.z() - in2p.z());
	return dist1 < dist2;
      }
  }

 private:
  edm::ESHandle<TrackerGeometry> tracker;

};

class LaserHitPairGeneratorFromLayerPair : public HitPairGenerator
{
 public:
  typedef CombinedHitPairGenerator::LayerCacheType LayerCacheType;

	/// constructor
  LaserHitPairGeneratorFromLayerPair(const LayerWithHits * inner, const LayerWithHits * outer, 
				     LayerCacheType * layerCache, const edm::EventSetup & iSetup) : theLayerCache(*layerCache),
    theInnerLayer(inner),theOuterLayer(outer) {}

	/// destructor
  virtual ~LaserHitPairGeneratorFromLayerPair() {}

	/// order hit pairs
  virtual OrderedHitPairs hitPairs(const TrackingRegion & region, const edm::EventSetup & iSetup)
  {
    return HitPairGenerator::hitPairs(region, iSetup);
  }

	/// from base class
  virtual void hitPairs(const TrackingRegion & ar, OrderedHitPairs & ap, const edm::EventSetup & iSetup);
  virtual LaserHitPairGeneratorFromLayerPair * clone() const 
  {
    return new LaserHitPairGeneratorFromLayerPair(*this);
  }

	/// return inner layer
  const LayerWithHits * innerLayer() const { return theInnerLayer; }
	/// return outer layer
  const LayerWithHits * outerLayer() const { return theOuterLayer; }

 private:
  // all data members are "shallow copy" 
  LayerCacheType & theLayerCache;
  const LayerWithHits * theInnerLayer;
  const LayerWithHits * theOuterLayer;
  const DetLayer * innerlay;
  const DetLayer * outerlay;

};

#endif
