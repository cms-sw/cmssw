#ifndef LaserAlignment_LaserLayerPairs_h
#define LaserAlignment_LaserLayerPairs_h

/** \class LaserLayerPairs
 *  find all (resonable) pairs of endcap layers
 *
 *  $Date: 2007/10/19 08:42:13 $
 *  $Revision: 1.5 $
 *  \author Maarten Thomas
 */

#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"

#include "Alignment/LaserAlignment/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h" 

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

class LaserLayerPairs : public SeedLayerPairs
{
 public:
	/// constructor
  LaserLayerPairs() : SeedLayerPairs() {};
	/// () operator
  std::vector<LayerPair> operator()();

	/// initialize layer pair finder
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
  SiStripRecHit2DCollection::range rphi_pos_range7;
  SiStripRecHit2DCollection::range rphi_pos_range8;
  SiStripRecHit2DCollection::range rphi_pos_range9;

  SiStripRecHit2DCollection::range rphi_neg_range1;
  SiStripRecHit2DCollection::range rphi_neg_range2;
  SiStripRecHit2DCollection::range rphi_neg_range3;
  SiStripRecHit2DCollection::range rphi_neg_range4;
  SiStripRecHit2DCollection::range rphi_neg_range5;
  SiStripRecHit2DCollection::range rphi_neg_range6;
  SiStripRecHit2DCollection::range rphi_neg_range7;
  SiStripRecHit2DCollection::range rphi_neg_range8;
  SiStripRecHit2DCollection::range rphi_neg_range9;

  TrackerLayerIdAccessor acc;

  LayerWithHits * lh1pos;
  LayerWithHits * lh2pos;
  LayerWithHits * lh3pos;
  LayerWithHits * lh4pos;
  LayerWithHits * lh5pos;
  LayerWithHits * lh6pos;
  LayerWithHits * lh7pos;
  LayerWithHits * lh8pos;
  LayerWithHits * lh9pos;

  LayerWithHits * lh1neg;
  LayerWithHits * lh2neg;
  LayerWithHits * lh3neg;
  LayerWithHits * lh4neg;
  LayerWithHits * lh5neg;
  LayerWithHits * lh6neg;
  LayerWithHits * lh7neg;
  LayerWithHits * lh8neg;
  LayerWithHits * lh9neg;

  std::vector<ForwardDetLayer*> fpos;
  std::vector<ForwardDetLayer*> fneg;

};

#endif
