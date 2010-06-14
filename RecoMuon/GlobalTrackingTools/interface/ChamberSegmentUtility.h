#ifndef RecoMuon_GlobalTrackingTools_ChamberSegmentUtility_h
#define RecoMuon_GlobalTrackingTools_ChamberSegmentUtility_h

/**
 *  Class: ChamberSegmentUtility
 *
 *  Description:
 *  utility class for the dynamical truncation algorithm
 *
 *  $Date: 2010/05/10 14:23:50 $
 *  $Revision: 1.1 $
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


using namespace std;
using namespace edm;

class ChamberSegmentUtility {

 public:

  ChamberSegmentUtility(const Event&, const edm::EventSetup&);

  // Get the 4D segments in a CSC chamber
  vector<CSCSegment> getCSCSegmentsInChamber(CSCDetId);

  // Get the 4D segments in a DT chamber
  vector<DTRecSegment4D> getDTSegmentsInChamber(DTChamberId);

  // Get the 1D rechits from a DTRecSegment4D
  vector<DTRecHit1D> getDTRHmap(DTRecSegment4D);

  // Get the 2D rechits from a CSCSegment 
  vector<CSCRecHit2D> getCSCRHmap(CSCSegment);

  
 private:

  ESHandle<CSCGeometry> cscGeometry;
  Handle<CSCSegmentCollection> CSCSegments;
  ESHandle<DTGeometry> dtGeom;
  Handle<DTRecSegment4DCollection> all4DSegments;

  vector<DTRecSegment4D> dtseg;
  vector<CSCSegment> cscseg;
  DTChamberId selectedDT;
  CSCDetId selectedCSC;
  vector<DTRecHit1D> phiSegRH;
  vector<DTRecHit1D> zSegRH;  
};

#endif


