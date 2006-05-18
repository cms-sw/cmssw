/**  \class GlobalMuonReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: $
 *   $Revision: $
 *
 *   \author   N. Neumeister            Purdue University
 *   \author   I. Belotelov             DUBNA
 **/

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonReFitter.h"

#include <iostream>
                                                                                

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

///constructor
GlobalMuonReFitter::GlobalMuonReFitter(const MagneticField * field) {

  theErrorRescaling = 100.0;
 
  thePropagator1 = new SteppingHelixPropagator(field,alongMomentum);
  thePropagator2 = new SteppingHelixPropagator(field,alongMomentum);
  
  theUpdator     = new KFUpdator;
  theEstimator   = new Chi2MeasurementEstimator(200.0);

}

///destructor
GlobalMuonReFitter::~GlobalMuonReFitter() {

  delete thePropagator2;
  delete thePropagator1;
  delete theUpdator;
  delete theEstimator;

}


vector<Trajectory> GlobalMuonReFitter::trajectories(const Trajectory& t) const {

  if ( !t.isValid() ) return vector<Trajectory>();

  vector<Trajectory> fitted = fit(t);
  return smooth(fitted);

}


vector<Trajectory> GlobalMuonReFitter::trajectories(const TrajectorySeed& seed,
	                                      const edm::OwnVector<TransientTrackingRecHit>& hits, 
	                                      const TrajectoryStateOnSurface& firstPredTsos) const  {

  if ( hits.empty() ) return vector<Trajectory>();

  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstPredTsos);
  vector<Trajectory> fitted = fit(seed, hits, firstTsos);

  return smooth(fitted);

}

vector<Trajectory> GlobalMuonReFitter::fit(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();
 
  TM firstTM = t.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fit(t.seed(), t.recHits(), firstTsos);

}


vector<Trajectory> GlobalMuonReFitter::fit(const TrajectorySeed& seed,
			             const edm::OwnVector<TransientTrackingRecHit>& hits, 
				     const TSOS& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();

  Trajectory myTraj(seed, thePropagator1->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if ( !predTsos.isValid() ) return vector<Trajectory>();
 
  TSOS currTsos;

  if ( hits.begin()->isValid() ) {
    // update
    currTsos = theUpdator->update(predTsos, (*hits.begin()));
    myTraj.push(TM(predTsos, currTsos, &(*hits.begin()),
		   theEstimator->estimate(predTsos, (*hits.begin())).second));
  } else {
    currTsos = predTsos;
    myTraj.push(TM(predTsos, &(*hits.begin())));
  }
  
  for ( edm::OwnVector<TransientTrackingRecHit>::const_iterator ihit = hits.begin() + 1; 
        ihit != hits.end(); ++ihit ) {

    predTsos = thePropagator1->propagate(currTsos, (*ihit).det()->surface());

    if ( !predTsos.isValid() ) {
      return vector<Trajectory>();
    } else if ( (*ihit).isValid() ) {
      // update
      currTsos = theUpdator->update(predTsos, *ihit);
      myTraj.push(TM(predTsos, currTsos, &(*ihit),
		     theEstimator->estimate(predTsos, *ihit).second));
    } else {
      currTsos = predTsos;
      myTraj.push(TM(predTsos, &(*ihit)));
    }
  }
  
  return vector<Trajectory>(1, myTraj);

}


vector<Trajectory> GlobalMuonReFitter::smooth(vector<Trajectory>& tc) const {

  vector<Trajectory> result; 
  
  for ( vector<Trajectory>::iterator it = tc.begin(); it != tc.end(); ++it ) {
    vector<Trajectory> smoothed = smooth(*it);
    result.insert(result.end(), smoothed.begin(), smoothed.end());
  }

  return result;

}


vector<Trajectory> GlobalMuonReFitter::smooth(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  Trajectory myTraj(t.seed(), thePropagator2->propagationDirection());

  vector<TM> avtm = t.measurements();
  if ( avtm.size() <= 2 ) return vector<Trajectory>(); 

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
		   avtm.back().estimate(),
		   avtm.back().layer()), 
		avtm.back().estimate());
  } else {
    currTsos = predTsos;
    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   avtm.back().recHit(),
                   avtm.back().estimate(),
		   avtm.back().layer()));
  }
  
  TrajectoryStateCombiner combiner;

  for ( vector<TM>::reverse_iterator itm = avtm.rbegin() + 1; 
        itm != avtm.rend() - 1; ++itm ) {

    predTsos = thePropagator2->propagate(currTsos,(*itm).recHit()->det()->surface());

    if ( !predTsos.isValid() ) {
      return vector<Trajectory>();
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
		     theEstimator->estimate(combTsos, (*(*itm).recHit())).second,
		     (*itm).layer()),
		  (*itm).estimate());
    } else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if ( !combTsos.isValid() ) return vector<Trajectory>();

      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     combTsos,
		     (*itm).recHit(),
                     (*itm).estimate(),
		     (*itm).layer()));
    }
  }
  
  // last smoothed TM is last filtered
  predTsos = thePropagator2->propagate(currTsos, avtm.front().recHit()->det()->surface());
  
  if ( !predTsos.isValid() ) return vector<Trajectory>();
  
  if ( avtm.front().recHit()->isValid() ) {
    //update
    currTsos = theUpdator->update(predTsos, (*avtm.front().recHit()));
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   theEstimator->estimate(predTsos, (*avtm.front().recHit())).second,
		   avtm.front().layer()),
		avtm.front().estimate());
  } else {
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   avtm.front().recHit(),
                   avtm.front().estimate(),
		   avtm.front().layer()));
  }

  return vector<Trajectory>(1, myTraj); 

}


