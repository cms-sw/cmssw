/* 
 * find all (resonable) pairs of endcap layers
 */

#ifndef LaserAlignment_LaserLayerPairs_h
#define LaserAlignment_LaserLayerPairs_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"

#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkDetLayers/interface/TECLayer.h" 

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

class LaserLayerPairs : public SeedLayerPairs
{
 public:
  LaserLayerPairs() : SeedLayerPairs() {};
  std::vector<LayerPair> operator()();

  void init(const SiStripRecHit2DCollection & collstereo, 
	    const SiStripRecHit2DCollection & collrphi,
	    const SiStripMatchedRecHit2DCollection & collmatched,
	    const edm::EventSetup & iSetup);

 private:
  // definition of the map
  SiStripRecHit2DCollection::range rphi_pos_range1;
  SiStripRecHit2DCollection::range rphi_pos_range2;
  SiStripRecHit2DCollection::range rphi_pos_range3;
  SiStripRecHit2DCollection::range rphi_pos_range4;
  SiStripRecHit2DCollection::range rphi_pos_range5;
  SiStripRecHit2DCollection::range rphi_pos_range6;

  SiStripRecHit2DCollection::range rphi_neg_range1;
  SiStripRecHit2DCollection::range rphi_neg_range2;
  SiStripRecHit2DCollection::range rphi_neg_range3;
  SiStripRecHit2DCollection::range rphi_neg_range4;
  SiStripRecHit2DCollection::range rphi_neg_range5;
  SiStripRecHit2DCollection::range rphi_neg_range6;

  TrackerLayerIdAccessor acc;

  LayerWithHits * lh1pos;
  LayerWithHits * lh2pos;
  LayerWithHits * lh3pos;
  LayerWithHits * lh4pos;
  LayerWithHits * lh5pos;
  LayerWithHits * lh6pos;

  LayerWithHits * lh1neg;
  LayerWithHits * lh2neg;
  LayerWithHits * lh3neg;
  LayerWithHits * lh4neg;
  LayerWithHits * lh5neg;
  LayerWithHits * lh6neg;

  std::vector<ForwardDetLayer*> fpos;
  std::vector<ForwardDetLayer*> fneg;

};

#endif
