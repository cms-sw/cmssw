/**  \class MuonTrackReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: 2007/04/12 00:39:16 $
 *   $Revision: 1.12 $
 *
 *   \author   N. Neumeister            Purdue University
 *   \author   C. Liu                   Purdue University
 **/

#include "RecoMuon/TrackingTools/interface/MuonTrackReFitter.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

//
// constructor
//
MuonTrackReFitter::MuonTrackReFitter(const ParameterSet& par, const MuonServiceProxy *service) : theService(service) {

  theErrorRescaling = 100.0;
 
  theUpdator     = new KFUpdator;
  theEstimator   = new Chi2MeasurementEstimator(200.0);
 
  theAlongMomentumProp = par.getParameter<string>("AlongMomentumPropagator");
  theOppositeToMomentumProp = par.getParameter<string>("OppositeToMomentumPropagator");
}

//
// destructor
//
MuonTrackReFitter::~MuonTrackReFitter() {

  if ( theUpdator ) delete theUpdator;
  if ( theEstimator ) delete theEstimator;

}

//
// fit and smooth trajectory
//
vector<Trajectory> MuonTrackReFitter::trajectories(const Trajectory& t) const {

  if ( !t.isValid() ) return vector<Trajectory>();

  vector<Trajectory> fitted = fit(t);
  return smooth(fitted);
  
}


//
// fit and smooth trajectory 
//
vector<Trajectory> MuonTrackReFitter::trajectories(const TrajectorySeed& seed,
	                                           const ConstRecHitContainer& hits, 
	                                           const TrajectoryStateOnSurface& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();

  //TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstPredTsos);
  TSOS firstTsos = firstPredTsos;
  firstTsos.rescaleError(10.);

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);
  return smooth(fitted);

}

//
// fit (not smooth) trajectory
//
vector<Trajectory> MuonTrackReFitter::trajectory(const Trajectory& t) const {

  if ( !t.isValid() ) return vector<Trajectory>();

  vector<Trajectory> fitted = fit(t);
  return fitted;

}


//
// fit (not smooth) trajectory 
//
vector<Trajectory> MuonTrackReFitter::trajectory(const TrajectorySeed& seed,
	                                           const ConstRecHitContainer& hits, 
	                                           const TrajectoryStateOnSurface& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();

  //TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstPredTsos);
  TSOS firstTsos = firstPredTsos;
  firstTsos.rescaleError(10.);

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);
  return fitted;

}


//
// fit trajectory
//
vector<Trajectory> MuonTrackReFitter::fit(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  TM firstTM = t.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());

  return fit(t.seed(), t.recHits(), firstTsos);

}


//
// fit trajectory
//
vector<Trajectory> MuonTrackReFitter::fit(const TrajectorySeed& seed,
			                  const ConstRecHitContainer& hits, 
				          const TSOS& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();

  Trajectory myTraj(seed, alongMomentum);

  TSOS predTsos(firstPredTsos);
  if ( !predTsos.isValid() ) return vector<Trajectory>();

  TSOS currTsos;

  if ( hits.front()->isValid() ) {
    // update
    TransientTrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);
    currTsos = theUpdator->update(predTsos, *preciseHit);
    myTraj.push(TM(predTsos, currTsos, hits.front(),
                   theEstimator->estimate(predTsos, *hits.front()).second));
  } else {
    currTsos = predTsos;
    myTraj.push(TM(predTsos, hits.front()));
  }
  //const TransientTrackingRecHit& firsthit = *hits.front();

  for ( ConstRecHitContainer::const_iterator ihit = hits.begin() + 1; 
        ihit != hits.end(); ++ihit ) {

    if ((**ihit).isValid() == false && (**ihit).det() == 0) continue;
    
    predTsos = theService->propagator(theAlongMomentumProp)->propagate(currTsos, (**ihit).det()->surface());

    if ( !predTsos.isValid() ) {
      //return vector<Trajectory>();
    } else if ( (**ihit).isValid() ) {
      // update
      TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);

      if (preciseHit->isValid() == false) {
        currTsos = predTsos;
        myTraj.push(TM(predTsos, *ihit));
      } else {
        currTsos = theUpdator->update(predTsos, *preciseHit);
        myTraj.push(TM(predTsos, currTsos, preciseHit,
                       theEstimator->estimate(predTsos, *preciseHit).second));
      }
    } else {
      currTsos = predTsos;
      myTraj.push(TM(predTsos, *ihit));
    }

  }

  return vector<Trajectory>(1, myTraj);

}


//
// smooth trajectory
//
vector<Trajectory> MuonTrackReFitter::smooth(const vector<Trajectory>& tc) const {

  vector<Trajectory> result; 

  for ( vector<Trajectory>::const_iterator it = tc.begin(); it != tc.end(); ++it ) {
    vector<Trajectory> smoothed = smooth(*it);
    result.insert(result.end(), smoothed.begin(), smoothed.end());
  }

  return result;

}


//
// smooth trajectory
//
vector<Trajectory> MuonTrackReFitter::smooth(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  Trajectory myTraj(t.seed(), oppositeToMomentum);

  vector<TM> avtm = t.measurements();
  if ( avtm.size() < 2 ) return vector<Trajectory>();

  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);

  if ( !predTsos.isValid() ) return vector<Trajectory>();

  TSOS currTsos;

  // first smoothed TM is last fitted
  if ( avtm.back().recHit()->isValid() ) {
    currTsos = theUpdator->update(predTsos, (*avtm.back().recHit()));
    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   predTsos,
		   avtm.back().updatedState(),
		   avtm.back().recHit(),
		   avtm.back().estimate()//,
		   /*avtm.back().layer()*/), 
	        avtm.back().estimate());

  } else {
    currTsos = predTsos;
    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   avtm.back().recHit()//,
		   /*avtm.back().layer()*/));

  }

  TrajectoryStateCombiner combiner;

  for ( vector<TM>::reverse_iterator itm = avtm.rbegin() + 1; 
        itm != avtm.rend() - 1; ++itm ) {

    predTsos = theService->propagator(theOppositeToMomentumProp)->propagate(currTsos,(*itm).recHit()->det()->surface());

    if ( !predTsos.isValid() ) {
      //return vector<Trajectory>();
    } else if ( (*itm).recHit()->isValid() ) {
      //update
      currTsos = theUpdator->update(predTsos, (*(*itm).recHit()));
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      if ( !combTsos.isValid() ) return vector<Trajectory>();

      TSOS smooTsos = combiner((*itm).updatedState(), predTsos);

      if ( !smooTsos.isValid() ) return vector<Trajectory>();

      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     smooTsos,
		     (*itm).recHit(),
		     theEstimator->estimate(combTsos, (*(*itm).recHit())).second//,
		     /*(*itm).layer()*/),
		     (*itm).estimate());
    } else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if ( !combTsos.isValid() ) return vector<Trajectory>();

      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     combTsos,
		     (*itm).recHit()//,
		     /*(*itm).layer()*/));
    }
  }

  // last smoothed TM is last filtered
  predTsos = theService->propagator(theOppositeToMomentumProp)->propagate(currTsos, avtm.front().recHit()->det()->surface());
  
  if ( !predTsos.isValid() ) return vector<Trajectory>();

  if ( avtm.front().recHit()->isValid() ) {
    //update
    currTsos = theUpdator->update(predTsos, (*avtm.front().recHit()));
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   theEstimator->estimate(predTsos, (*avtm.front().recHit())).second//,
		   /*avtm.front().layer()*/),
	        avtm.front().estimate());
  } else {
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   avtm.front().recHit()//,
		   /*avtm.front().layer()*/));
  }

  if (myTraj.foundHits() > 3)
    return vector<Trajectory>(1, myTraj);
  else return vector<Trajectory>();

}

