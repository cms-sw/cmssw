#ifndef EstimatorTool_RecSegment_h
#define EstimatorTool_RecSegment_h

/**
 *  Class: ChamberSegmentUtility
 *
 *  Description:
 *  utility class for the dynamical truncation algorithm
 *
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"


class ChamberSegmentUtility {

 public:

  ChamberSegmentUtility();

  void initCSU(const edm::Handle<DTRecSegment4DCollection>&, const edm::Handle<CSCSegmentCollection>&);

  // Get the 4D segments in a CSC chamber
  std::vector<CSCSegment> getCSCSegmentsInChamber(CSCDetId);

  // Get the 4D segments in a DT chamber
  std::vector<DTRecSegment4D> getDTSegmentsInChamber(DTChamberId);
  
 private:

  edm::ESHandle<CSCGeometry> cscGeometry;
  edm::Handle<CSCSegmentCollection> CSCSegments;
  edm::ESHandle<DTGeometry> dtGeom;
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  std::vector<DTRecSegment4D> dtseg;
  std::vector<CSCSegment> cscseg;
};

#endif


