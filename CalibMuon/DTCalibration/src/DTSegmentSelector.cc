#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

bool DTSegmentSelector::operator() (DTRecSegment4D const& segment, edm::Event const& event, edm::EventSetup const& setup){

  bool result = true;

  /*
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);
  */
 
  edm::ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels_) setup.get<DTStatusFlagRcd>().get(statusMap);

  // Get the Phi 2D segment
  int nPhiHits = -1;
  bool segmentNoisyPhi = false;
  if( segment.hasPhi() ){
     const DTChamberRecSegment2D* phiSeg = segment.phiSegment();  // phiSeg lives in the chamber RF
     //LocalPoint phiSeg2DPosInCham = phiSeg->localPosition();  
     //LocalVector phiSeg2DDirInCham = phiSeg->localDirection();
     std::vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();
     nPhiHits = phiRecHits.size(); 
     if(checkNoisyChannels_) segmentNoisyPhi = checkNoisySegment(statusMap,phiRecHits);
  }
  // Get the Theta 2D segment
  int nZHits = -1;
  bool segmentNoisyZ = false;
  if( segment.hasZed() ){
     const DTSLRecSegment2D* zSeg = segment.zSegment();  // zSeg lives in the SL RF
     //const DTSuperLayer* sl = chamber->superLayer(zSeg->superLayerId());
     //LocalPoint zSeg2DPosInCham = chamber->toLocal(sl->toGlobal((*zSeg).localPosition())); 
     //LocalVector zSeg2DDirInCham = chamber->toLocal(sl->toGlobal((*zSeg).localDirection()));
     std::vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
     nZHits = zRecHits.size();
     if(checkNoisyChannels_) segmentNoisyZ = checkNoisySegment(statusMap,zRecHits);
  } 

  // Segment selection 
  // Discard segment if it has a noisy cell
  if(segmentNoisyPhi || segmentNoisyZ)
     result = false;

  // 2D-segment number of hits
  if(segment.hasPhi() && nPhiHits < minHitsPhi_)
     result = false;

  if(segment.hasZed() && nZHits < minHitsZ_)
     result = false;

  // Segment chi2
  double chiSquare = segment.chi2()/segment.degreesOfFreedom();
  if(chiSquare > maxChi2_)
     result = false;

  // Segment angle
  LocalPoint segment4DLocalPos = segment.localPosition();
  LocalVector segment4DLocalDir = segment.localDirection();
  double angleZ = fabs( atan(segment4DLocalDir.y()/segment4DLocalDir.z())*180./Geom::pi() ); 
  if( angleZ > maxAngleZ_)
     result = false;

  double anglePhi = fabs( atan(segment4DLocalDir.x()/segment4DLocalDir.z())*180./Geom::pi() );
  if( anglePhi > maxAnglePhi_)
     result = false;

  return result;
}

bool DTSegmentSelector::checkNoisySegment(edm::ESHandle<DTStatusFlag> const& statusMap, std::vector<DTRecHit1D> const& dtHits){

  bool segmentNoisy = false;

  std::vector<DTRecHit1D>::const_iterator dtHit = dtHits.begin();
  std::vector<DTRecHit1D>::const_iterator dtHits_end = dtHits.end();
  for(; dtHit != dtHits_end; ++dtHit){
     //DTRecHit1D const* dtHit1D = dynamic_cast<DTRecHit1D const*>(*recHit);
     DTWireId wireId = dtHit->wireId();
     // Check for noisy channels to skip them
     bool isNoisy = false, isFEMasked = false, isTDCMasked = false, isTrigMask = false,
          isDead = false, isNohv = false;
     statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
     if(isNoisy) {
        LogTrace("Calibration") << "Wire: " << wireId << " is noisy, skipping!";
        segmentNoisy = true; break;
     }
  }
  return segmentNoisy;
}
