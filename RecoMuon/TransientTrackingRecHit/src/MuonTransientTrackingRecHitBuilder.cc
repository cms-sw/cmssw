/**
 *  Class: MuonTransientTrackingRecHitBuilder
 *
 *  Description:
 *
 *
 *  $Date: 2006/09/01 15:48:56 $
 *  $Revision: 1.4 $
 *
 *  Authors :
 *  A. Everett               Purdue University
 *
 **/

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

MuonTransientTrackingRecHitBuilder::MuonTransientTrackingRecHitBuilder(edm::ESHandle<GlobalTrackingGeometry> trackingGeometry):
  theTrackingGeometry(trackingGeometry)
{}


MuonTransientTrackingRecHitBuilder::RecHitPointer
MuonTransientTrackingRecHitBuilder::build (const TrackingRecHit* p, 
					   edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  
  if ( p->geographicalId().det() == DetId::Muon ) {
    return (MuonTransientTrackingRecHit::specificBuild(trackingGeometry->idToDet(p->geographicalId()),p).get());
  }
  
  return 0;

}

MuonTransientTrackingRecHitBuilder::RecHitPointer
MuonTransientTrackingRecHitBuilder::build(const TrackingRecHit * p) const{
  if(theTrackingGeometry.isValid()) return build(p,theTrackingGeometry);
  else
    throw cms::Exception("Muon|RecoMuon|MuonTransientTrackingRecHitBuilder")
      <<"ERROR! You are trying to build a MuonTransientTrackingRecHit with a non valid GlobalTrackingGeometry";
}
