/** \file CosmicMuonSmoother
 *
 *  The class refit muon trajectories based on MuonTrackReFitter
 *  But 
 *  (1) check direction to make sure it's downward for cosmic
 *  (2) propagate by virtual intermediate planes if failed to ensure propagation
 *      within cylinders
 *
 *
 *  $Date: 2007/03/27 20:49:45 $
 *  $Revision: 1.5 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

//
// constructor
//
CosmicMuonSmoother::CosmicMuonSmoother(const ParameterSet& par, const MuonServiceProxy *service) : theService(service) {

  theUpdator     = new KFUpdator;
  theUtilities   = new CosmicMuonUtilities; 
  theEstimator   = new Chi2MeasurementEstimator(200.0);
  thePropagatorName = par.getParameter<string>("Propagator");

}

//
// destructor
//
CosmicMuonSmoother::~CosmicMuonSmoother() {

  if ( theUpdator ) delete theUpdator;
  if ( theUtilities ) delete theUtilities;
  if ( theEstimator ) delete theEstimator;

}


//
// fit and smooth trajectory
//
vector<Trajectory> CosmicMuonSmoother::trajectories(const Trajectory& t) const {
   vector<Trajectory> fitted = fit(t);
   return smooth(fitted);

}

//
// fit and smooth trajectory 
//
vector<Trajectory> CosmicMuonSmoother::trajectories(const TrajectorySeed& seed,
	                                           const ConstRecHitContainer& hits, 
	                                           const TrajectoryStateOnSurface& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();

  TrajectoryStateOnSurface firstTsos = firstPredTsos;
  firstTsos.rescaleError(10.);

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);

  return smooth(fitted);

}

//
// fit trajectory
//
vector<Trajectory> CosmicMuonSmoother::fit(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  TrajectoryMeasurement firstTM = t.firstMeasurement();
  TrajectoryMeasurement lastTM = t.lastMeasurement();

  TrajectoryStateOnSurface firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());

  TrajectoryStateOnSurface lastTsos = TrajectoryStateWithArbitraryError()(lastTM.updatedState());

  if (firstTsos.globalPosition().y() < firstTsos.globalPosition().y())
        firstTsos = lastTsos;

  if (firstTsos.globalMomentum().y()> 1.0 && firstTsos.globalMomentum().eta()< 4.0 ) 
     theUtilities->reverseDirection(firstTsos,&*theService->magneticField());

  ConstRecHitContainer hits = t.recHits();

  // sort RecHits AlongMomentum
  if (hits.front()->globalPosition().y() < hits.back()->globalPosition().y())
      std::reverse(hits.begin(),hits.end());

  return fit(t.seed(), t.recHits(), firstTsos);

}


//
// fit trajectory
//
vector<Trajectory> CosmicMuonSmoother::fit(const TrajectorySeed& seed,
			                  const ConstRecHitContainer& hits, 
				          const TrajectoryStateOnSurface& firstPredTsos) const {

  if ( hits.empty() ) return vector<Trajectory>();


  Trajectory myTraj(seed, alongMomentum);

  TrajectoryStateOnSurface predTsos(firstPredTsos);
  if ( !predTsos.isValid() ) return vector<Trajectory>();

  TrajectoryStateOnSurface currTsos;

  if ( hits.front()->isValid() ) {

    TransientTrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);

    currTsos = theUpdator->update(predTsos, *preciseHit);
    myTraj.push(TrajectoryMeasurement(predTsos, currTsos, hits.front(),
                theEstimator->estimate(predTsos, *hits.front()).second));

  } else {

    currTsos = predTsos;
    myTraj.push(TrajectoryMeasurement(predTsos, hits.front()));
  }
  //const TransientTrackingRecHit& firsthit = *hits.front();

  for ( ConstRecHitContainer::const_iterator ihit = hits.begin() + 1; 
        ihit != hits.end(); ++ihit ) {

    if ((**ihit).isValid() == false && (**ihit).det() == 0) continue;

    predTsos = propagator()->propagate(currTsos, (**ihit).det()->surface());

    if ( !predTsos.isValid() ) {

       predTsos = theUtilities->stepPropagate(currTsos, (*ihit), *propagator());

    }

    if ( !predTsos.isValid() ) {

      //return vector<Trajectory>();
    } else if ( (**ihit).isValid() ) {
      // update
      TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);

      if (preciseHit->isValid() == false) {
        currTsos = predTsos;
        myTraj.push(TrajectoryMeasurement(predTsos, *ihit));
      } else {
        currTsos = theUpdator->update(predTsos, *preciseHit);
        myTraj.push(TrajectoryMeasurement(predTsos, currTsos, preciseHit,
                       theEstimator->estimate(predTsos, *preciseHit).second));
      }
    } else {
      currTsos = predTsos;
      myTraj.push(TrajectoryMeasurement(predTsos, *ihit));
    }

  }

  return vector<Trajectory>(1, myTraj);

}


//
// smooth trajectory
//
vector<Trajectory> CosmicMuonSmoother::smooth(const vector<Trajectory>& tc) const {

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
vector<Trajectory> CosmicMuonSmoother::smooth(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  Trajectory myTraj(t.seed(), oppositeToMomentum);

  vector<TrajectoryMeasurement> avtm = t.measurements();

  if ( avtm.size() < 2 ) return vector<Trajectory>();

  TrajectoryStateOnSurface predTsos = avtm.back().forwardPredictedState();
 // predTsos.rescaleError(theErrorRescaling);

  if ( !predTsos.isValid() ) return vector<Trajectory>();

  TrajectoryStateOnSurface currTsos;

  // first smoothed TrajectoryMeasurement is last fitted
  if ( avtm.back().recHit()->isValid() ) {
    currTsos = theUpdator->update(predTsos, (*avtm.back().recHit()));
    myTraj.push(TrajectoryMeasurement(avtm.back().forwardPredictedState(),
		   predTsos,
		   avtm.back().updatedState(),
		   avtm.back().recHit(),
		   avtm.back().estimate()//,
		   /*avtm.back().layer()*/), 
	        avtm.back().estimate());

  } else {
    currTsos = predTsos;
    myTraj.push(TrajectoryMeasurement(avtm.back().forwardPredictedState(),
		   avtm.back().recHit()//,
		   /*avtm.back().layer()*/));

  }

  TrajectoryStateCombiner combiner;


  for ( vector<TrajectoryMeasurement>::reverse_iterator itm = avtm.rbegin() + 1; 
        itm != avtm.rend() - 1; ++itm ) {

    predTsos = propagator()->propagate(currTsos,(*itm).recHit()->det()->surface());

    if ( !predTsos.isValid() ) {
       predTsos = theUtilities->stepPropagate(currTsos, (*itm).recHit(), *propagator());
    }

    if ( !predTsos.isValid() ) {
      //return vector<Trajectory>();
    } else if ( (*itm).recHit()->isValid() ) {
      //update
      currTsos = theUpdator->update(predTsos, (*(*itm).recHit()));
      TrajectoryStateOnSurface combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      if ( !combTsos.isValid() ) return vector<Trajectory>();

      TrajectoryStateOnSurface smooTsos = combiner((*itm).updatedState(), predTsos);

      if ( !smooTsos.isValid() ) return vector<Trajectory>();

      myTraj.push(TrajectoryMeasurement((*itm).forwardPredictedState(),
		     predTsos,
		     smooTsos,
		     (*itm).recHit(),
		     theEstimator->estimate(combTsos, (*(*itm).recHit())).second//,
		     /*(*itm).layer()*/),
		     (*itm).estimate());
    } else {
      currTsos = predTsos;
      TrajectoryStateOnSurface combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if ( !combTsos.isValid() ) return vector<Trajectory>();

      myTraj.push(TrajectoryMeasurement((*itm).forwardPredictedState(),
		     predTsos,
		     combTsos,
		     (*itm).recHit()//,
		     /*(*itm).layer()*/));
    }
  }

  // last smoothed TrajectoryMeasurement is last filtered
  predTsos = propagator()->propagate(currTsos, avtm.front().recHit()->det()->surface());
  
  if ( !predTsos.isValid() ) return vector<Trajectory>();

  if ( avtm.front().recHit()->isValid() ) {
    //update
    currTsos = theUpdator->update(predTsos, (*avtm.front().recHit()));
    myTraj.push(TrajectoryMeasurement(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   theEstimator->estimate(predTsos, (*avtm.front().recHit())).second//,
		   /*avtm.front().layer()*/),
	        avtm.front().estimate());
  } else {
    myTraj.push(TrajectoryMeasurement(avtm.front().forwardPredictedState(),
		   avtm.front().recHit()//,
		   /*avtm.front().layer()*/));
  }

  if (myTraj.foundHits() > 3)
    return vector<Trajectory>(1, myTraj);
  else return vector<Trajectory>();

}
