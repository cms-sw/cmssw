/**
 *  Class: MuonTransientTrackingRecHitBuilder
 *
 *  Description:
 *
 *
 *  $Date: 2006/08/28 14:28:04 $
 *  $Revision: 1.3 $
 *
 *  Authors :
 *  A. Everett               Purdue University
 *
 **/

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

MuonTransientTrackingRecHitBuilder::RecHitPointer
MuonTransientTrackingRecHitBuilder::build (const TrackingRecHit* p, 
					   edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  
  if ( p->geographicalId().det() == DetId::Muon ) {
    return (MuonTransientTrackingRecHit::specificBuild(trackingGeometry->idToDet(p->geographicalId()),p).get());
  }
  
  return 0;

}
