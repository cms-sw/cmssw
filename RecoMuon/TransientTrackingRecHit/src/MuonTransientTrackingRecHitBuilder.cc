/**
 *  Class: MuonTransientTrackingRecHitBuilder
 *
 *  Description:
 *
 *
 *  $Date: $
 *  $Revision: $
 *
 *  Authors :
 *  A. Everett               Purdue University
 *
 **/

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

//
//
//
MuonTransientTrackingRecHitBuilder::MuonTransientTrackingRecHitBuilder(const edm::ParameterSet& ) {}


//
//
//
void MuonTransientTrackingRecHitBuilder::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

}


//
//
//
MuonTransientTrackingRecHitBuilder::RecHitPointer
MuonTransientTrackingRecHitBuilder::build (const TrackingRecHit * p) const {

  if ( p->geographicalId().det() == DetId::Muon ) {
    return (MuonTransientTrackingRecHit::specificBuild(theTrackingGeometry->idToDet(p->geographicalId()),p).get());
  }

  return 0;

}
