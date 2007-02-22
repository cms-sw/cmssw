/*
 * find all (resonable) pairs of layers
 */

#include "Alignment/LaserAlignment/interface/LaserLayerPairs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/SiStripDetId/interface/TECDetId.h"

std::vector<SeedLayerPairs::LayerPair> LaserLayerPairs::operator()()
{
  std::vector<LayerPair> result;

  // seeds from TEC+
  result.push_back( LayerPair(lh1pos, lh2pos) );
  result.push_back( LayerPair(lh1pos, lh3pos) );
  result.push_back( LayerPair(lh1pos, lh4pos) );
  result.push_back( LayerPair(lh1pos, lh5pos) );
  result.push_back( LayerPair(lh1pos, lh6pos) );

  result.push_back( LayerPair(lh2pos, lh3pos) );
  result.push_back( LayerPair(lh2pos, lh4pos) );
  result.push_back( LayerPair(lh2pos, lh5pos) );
  result.push_back( LayerPair(lh2pos, lh6pos) );

  result.push_back( LayerPair(lh3pos, lh4pos) );
  result.push_back( LayerPair(lh3pos, lh5pos) );
  result.push_back( LayerPair(lh3pos, lh6pos) );

  result.push_back( LayerPair(lh4pos, lh5pos) );
  result.push_back( LayerPair(lh4pos, lh6pos) );

  result.push_back( LayerPair(lh5pos, lh6pos) );

  // seeds from TEC-
  result.push_back( LayerPair(lh1neg, lh2neg) );
  result.push_back( LayerPair(lh1neg, lh3neg) );
  result.push_back( LayerPair(lh1neg, lh4neg) );
  result.push_back( LayerPair(lh1neg, lh5neg) );
  result.push_back( LayerPair(lh1neg, lh6neg) );

  result.push_back( LayerPair(lh2neg, lh3neg) );
  result.push_back( LayerPair(lh2neg, lh4neg) );
  result.push_back( LayerPair(lh2neg, lh5neg) );
  result.push_back( LayerPair(lh2neg, lh6neg) );

  result.push_back( LayerPair(lh3neg, lh4neg) );
  result.push_back( LayerPair(lh3neg, lh5neg) );
  result.push_back( LayerPair(lh3neg, lh6neg) );

  result.push_back( LayerPair(lh4neg, lh5neg) );
  result.push_back( LayerPair(lh4neg, lh6neg) );

  result.push_back( LayerPair(lh5neg, lh6neg) );

  return result;
}

void LaserLayerPairs::init(const SiStripRecHit2DCollection & collstereo,
			   const SiStripRecHit2DCollection & collrphi,
			   const SiStripMatchedRecHit2DCollection & collmatched,
			   const edm::EventSetup & iSetup)
{
  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get(track);

  // get the discs in TEC+
  fpos = track->posTecLayers();
  // get the discs in TEC-
  fneg = track->negTecLayers();

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  // side = 1 for backward, 2 for forward TEC
  rphi_pos_range1 = collrphi.get(acc.stripTECDisk(2,1)); // disc 1
  rphi_pos_range2 = collrphi.get(acc.stripTECDisk(2,2)); // disc 2
  rphi_pos_range3 = collrphi.get(acc.stripTECDisk(2,3)); // disc 3
  rphi_pos_range4 = collrphi.get(acc.stripTECDisk(2,4)); // disc 4
  rphi_pos_range5 = collrphi.get(acc.stripTECDisk(2,5)); // disc 5
  rphi_pos_range6 = collrphi.get(acc.stripTECDisk(2,6)); // disc 6

  rphi_neg_range1 = collrphi.get(acc.stripTECDisk(1,1)); // disc 1
  rphi_neg_range2 = collrphi.get(acc.stripTECDisk(1,2)); // disc 2
  rphi_neg_range3 = collrphi.get(acc.stripTECDisk(1,3)); // disc 3
  rphi_neg_range4 = collrphi.get(acc.stripTECDisk(1,4)); // disc 4
  rphi_neg_range5 = collrphi.get(acc.stripTECDisk(1,5)); // disc 5
  rphi_neg_range6 = collrphi.get(acc.stripTECDisk(1,6)); // disc 6

  // get the discs
  const TECLayer * fposl1 = dynamic_cast<TECLayer*>(fpos[0]);
  const TECLayer * fposl2 = dynamic_cast<TECLayer*>(fpos[1]);
  const TECLayer * fposl3 = dynamic_cast<TECLayer*>(fpos[2]);
  const TECLayer * fposl4 = dynamic_cast<TECLayer*>(fpos[3]);
  const TECLayer * fposl5 = dynamic_cast<TECLayer*>(fpos[4]);
  const TECLayer * fposl6 = dynamic_cast<TECLayer*>(fpos[5]);

  const TECLayer * fnegl1 = dynamic_cast<TECLayer*>(fneg[0]);
  const TECLayer * fnegl2 = dynamic_cast<TECLayer*>(fneg[1]);
  const TECLayer * fnegl3 = dynamic_cast<TECLayer*>(fneg[2]);
  const TECLayer * fnegl4 = dynamic_cast<TECLayer*>(fneg[3]);
  const TECLayer * fnegl5 = dynamic_cast<TECLayer*>(fneg[4]);
  const TECLayer * fnegl6 = dynamic_cast<TECLayer*>(fneg[5]);

  // Layers with hits
  lh1pos = new LayerWithHits(fposl1, rphi_pos_range1);
  lh2pos = new LayerWithHits(fposl2, rphi_pos_range2);
  lh3pos = new LayerWithHits(fposl3, rphi_pos_range3);
  lh4pos = new LayerWithHits(fposl4, rphi_pos_range4);
  lh5pos = new LayerWithHits(fposl5, rphi_pos_range5);
  lh6pos = new LayerWithHits(fposl6, rphi_pos_range6);

  lh1neg = new LayerWithHits(fnegl1, rphi_neg_range1);
  lh2neg = new LayerWithHits(fnegl2, rphi_neg_range2);
  lh3neg = new LayerWithHits(fnegl3, rphi_neg_range3);
  lh4neg = new LayerWithHits(fnegl4, rphi_neg_range4);
  lh5neg = new LayerWithHits(fnegl5, rphi_neg_range5);
  lh6neg = new LayerWithHits(fnegl6, rphi_neg_range6);
}
