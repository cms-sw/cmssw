#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

bool DTSegmentSelector::operator() (edm::Event const& event, edm::EventSetup const& setup, DTRecSegment4D const& segment){

  bool result = true;

  /*
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);
  */
 
  bool segmentNoisy = false;
  if(checkNoisyChannels_){
     // Get the map of noisy channels
     edm::ESHandle<DTStatusFlag> statusMap;
     setup.get<DTStatusFlagRcd>().get(statusMap);

     if( segment.hasPhi() ){
        const DTChamberRecSegment2D* phiSeg = segment.phiSegment();
        segmentNoisy = checkNoisySegment(statusMap,*phiSeg); 
     }
     if( segment.hasZed() ){
        const DTSLRecSegment2D* zSeg = segment.zSegment();
        segmentNoisy = checkNoisySegment(statusMap,*zSeg);
     }
  }
  if(segmentNoisy) result = false;

  /*
  // Get the Phi 2D segment
  if( segment.hasPhi() ){
     const DTChamberRecSegment2D* phiSeg = segment.phiSegment();  // phiSeg lives in the chamber RF
     LocalPoint phiSeg2DPosInCham = phiSeg->localPosition();  
     LocalVector phiSeg2DDirInCham = phiSeg->localDirection();
  }
  // Get the Theta 2D segment
  if( segment.hasZed() ){
     const DTSLRecSegment2D* zSeg = segment.zSegment();  // zSeg lives in the SL RF
     const DTSuperLayer* sl = chamber->superLayer(zSeg->superLayerId());
     LocalPoint zSeg2DPosInCham = chamber->toLocal(sl->toGlobal((*zSeg).localPosition())); 
     LocalVector zSeg2DDirInCham = chamber->toLocal(sl->toGlobal((*zSeg).localDirection()));
  } 
  */

  // Selection 
  // Get the segment chi2
  double chiSquare = segment.chi2()/segment.degreesOfFreedom();
  // Cut on the segment chi2 
  if(chiSquare > maxChi2_) result = false;
  LocalPoint segment4DLocalPos = segment.localPosition();
  LocalVector segment4DLocalDir = segment.localDirection();
  // Cut on angle
  if( fabs( atan(segment4DLocalDir.y()/segment4DLocalDir.z())*180./Geom::pi() ) > maxAngleZ_) 
     result = false;
  // Cut on angle 
  if( fabs( atan(segment4DLocalDir.x()/segment4DLocalDir.z())*180./Geom::pi() ) > maxAnglePhi_)
     result = false;

  return result;
}

template <class T>
bool DTSegmentSelector::checkNoisySegment(edm::ESHandle<DTStatusFlag> const& statusMap, T const& segment){

  bool segmentNoisy = false;

  std::vector<DTRecHit1D> const& recHits = segment.specificRecHits();
  std::vector<DTRecHit1D>::const_iterator recHit = recHits.begin();
  std::vector<DTRecHit1D>::const_iterator recHits_end = recHits.end();
  for(; recHit != recHits_end; ++recHit){
     DTWireId wireId = recHit->wireId();
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
