/**
 *  Class: TrackConverter
 *
 *  Description:
 *     Convert a reco::Track into a Trajectory by
 *     performing a refit
 *
 *
 *  $Date: 2006/08/28 20:40:17 $
 *  $Revision: 1.2 $ 
 *
 *  Authors :
 *  N. Neumeister            Purdue University
 *  A. Everett               Purdue University
 *
 **/

#include "RecoMuon/TrackingTools/interface/TrackConverter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackReFitter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

//
// constructor
//
TrackConverter::TrackConverter(const edm::ParameterSet& par) {

  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new MuonTrackReFitter(refitterPSet);

}


//
// destructor
//
TrackConverter::~TrackConverter() {

  if ( theRefitter ) delete theRefitter;

}


//
// percolate the Event Setup
//
void TrackConverter::setES(const edm::EventSetup& setup) {

  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  theRefitter->setES(setup);

}


//
// set the TransientTrackingRecHitBuilder
//
void TrackConverter::setBuilder(TransientTrackingRecHitBuilder* tkTTRHB,MuonTransientTrackingRecHitBuilder* muTTRHB) {

  theTkHitBuilder = tkTTRHB;
  theMuHitBuilder = muTTRHB;

}


//
// convert a reco::TrackRef into a Trajectory
//
vector<Trajectory> TrackConverter::convert(const reco::TrackRef& t) const {

  vector<Trajectory> result  = convert(*t);

  return result;

}


//
// convert a reco::Track into a Trajectory
//
vector<Trajectory> TrackConverter::convert(const reco::Track& t) const {
  
  vector<Trajectory> result;
  
  // use TransientTrackingRecHitBuilder to get TransientTrackingRecHits
  ConstRecHitContainer hits = getTransientRecHits(t);
  
  // sort RecHits AlongMomentum
  if ( hits.front()->geographicalId().det() == DetId::Tracker ) {
    reverse(hits.begin(),hits.end());
  }
  //printHits(hits);

  // use TransientTrackBuilder to get a starting TSOS
  reco::TransientTrack theTT(t,&*theMagField,theTrackingGeometry);
  TrajectoryStateOnSurface firstState = theTT.innermostMeasurementState();
  if ( hits.front()->geographicalId().det() == DetId::Tracker ) {
    firstState = theRefitter->propagator()->propagate(theTT.impactPointState(), hits.front()->det()->surface());
  }
  else {
    firstState = theTT.innermostMeasurementState();
  }

  //cout << "INNER: " << firstState.globalPosition().perp() << " " <<  firstState.globalPosition().z() << " " << firstState.globalMomentum() << endl;

  AlgebraicSymMatrix C(5,1);
  C *= 10.;
  LocalTrajectoryParameters lp = firstState.localParameters();
  TrajectoryStateOnSurface theTSOS(lp,LocalTrajectoryError(C),
                                   firstState.surface(), 
                                   &*theMagField);
  
  theTSOS = firstState;
// theTSOS.rescaleError(3.);

  const TrajectorySeed* seed = new TrajectorySeed();
  vector<Trajectory> trajs = theRefitter->trajectories(*seed,hits,theTSOS);
 
  if ( !trajs.empty()) result.push_back(trajs.front());
 
  return result;

}


//
// get container of transient RecHits from a Track
//
TrackConverter::ConstRecHitContainer 
TrackConverter::getTransientRecHits(const reco::Track& track) const {

   ConstRecHitContainer result;

   for (trackingRecHit_iterator iter = track.recHitsBegin(); iter != track.recHitsEnd(); ++iter) {;
     if ( (*iter)->geographicalId().det() != DetId::Muon ) {
       result.push_back(theTkHitBuilder->build(&**iter));
     }
     else {
       result.push_back(theMuHitBuilder->build(&**iter));
     }
   }

   return result;

}


//
// get container of transient muon RecHits from a Track
//
TrackConverter::ConstMuonRecHitContainer
TrackConverter::getTransientMuonRecHits(const reco::Track& track) const {

   ConstMuonRecHitContainer result;
   for (trackingRecHit_iterator iter = track.recHitsBegin(); iter != track.recHitsEnd(); ++iter) {

     const TrackingRecHit* p = (*iter).get();
     const GeomDet* gd = theTrackingGeometry->idToDet(p->geographicalId());

      MuonTransientTrackingRecHit::MuonRecHitPointer mp = MuonTransientTrackingRecHit::specificBuild(gd,p);

     result.push_back(mp);

   }

   return result;

}


//
// print RecHits
//
void TrackConverter::printHits(const ConstRecHitContainer& hits) const {

  LogInfo("GlobalMuonTrajectoryBuilder") << "Used RecHits: ";
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogInfo("GlobalMuonTrajectoryBuilder") << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogInfo("GlobalMuonTrajectoryBuilder")
    << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())     << "  z = " << pos.z()
    << "  dimension = " << (*ir)->dimension()
    << "  " << (*ir)->det()->geographicalId().det()
    << "  " << (*ir)->det()->subDetector();
  }

}
