/**
 *  Class: MuonTransientTrackingRecHitBuilder
 *
 *  Description:
 *
 *
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
    return MuonTransientTrackingRecHit::specificBuild(trackingGeometry->idToDet(p->geographicalId()),p);
  }
  
  return RecHitPointer();

}

MuonTransientTrackingRecHitBuilder::RecHitPointer
MuonTransientTrackingRecHitBuilder::build(const TrackingRecHit * p) const {
  if(theTrackingGeometry.isValid()) return build(p,theTrackingGeometry);
  else
    throw cms::Exception("Muon|RecoMuon|MuonTransientTrackingRecHitBuilder")
      <<"ERROR! You are trying to build a MuonTransientTrackingRecHit with a non valid GlobalTrackingGeometry";
}

MuonTransientTrackingRecHitBuilder::ConstRecHitContainer 
MuonTransientTrackingRecHitBuilder::build(const trackingRecHit_iterator& start, const trackingRecHit_iterator& stop) const {
 
  ConstRecHitContainer result;
  for(trackingRecHit_iterator hit = start; hit != stop; ++hit )
    result.push_back(build(&**hit));
  
  return result;
}
  
