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

#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <map>
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;

ChamberSegmentUtility::ChamberSegmentUtility() {}

void ChamberSegmentUtility::initCSU(const edm::Handle<DTRecSegment4DCollection>& DTSegProd,
                                    const edm::Handle<CSCSegmentCollection>& CSCSegProd) {
  all4DSegments = DTSegProd;
  CSCSegments = CSCSegProd;
}

vector<CSCSegment> ChamberSegmentUtility::getCSCSegmentsInChamber(CSCDetId sel) {
  cscseg.clear();
  CSCSegmentCollection::range range = CSCSegments->get(sel);
  for (CSCSegmentCollection::const_iterator segment = range.first; segment != range.second; ++segment) {
    cscseg.push_back(*segment);
  }
  return cscseg;
}

vector<DTRecSegment4D> ChamberSegmentUtility::getDTSegmentsInChamber(DTChamberId sel) {
  dtseg.clear();
  DTRecSegment4DCollection::range range = all4DSegments->get(sel);
  for (DTRecSegment4DCollection::const_iterator segment = range.first; segment != range.second; ++segment) {
    dtseg.push_back(*segment);
  }
  return dtseg;
}
