/** \file CosmicMuonSmoother
 *
 *  The class refit muon trajectories based on MuonTrackReFitter
 *  But 
 *  (1) check direction to make sure it's downward for cosmic
 *  (2) propagate by virtual intermediate planes if failed to ensure propagation
 *      within cylinders
 *
 *
 *  $Date: 2007/08/16 20:00:23 $
 *  $Revision: 1.7 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

  category_ = "Muon|RecoMuon|CosmicMuon|CosmicMuonSmoother";

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

  if ( hits.empty() ||!firstPredTsos.isValid() ) return vector<Trajectory>();

  LogTrace(category_)<< "trajectory begin (seed hits tsos)";

  TrajectoryStateOnSurface firstTsos = firstPredTsos;
  firstTsos.rescaleError(20.);

  LogTrace(category_)<< "first TSOS: "<<firstTsos;

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);

  return smooth(fitted);

}

//
// fit trajectory
//
vector<Trajectory> CosmicMuonSmoother::fit(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  LogTrace(category_)<< "fit begin (trajectory) ";

  TrajectoryStateOnSurface firstTsos = initialState(t); 
  if ( !firstTsos.isValid() ) {
    LogTrace(category_)<< "Error: firstTsos invalid. ";
    return vector<Trajectory>();
  }

  LogTrace(category_)<< "firstTsos: "<<firstTsos;

  ConstRecHitContainer hits = t.recHits();
  LogTrace(category_)<< "hits: "<<hits.size();

  sortHitsAlongMom(hits, firstTsos);

  return fit(t.seed(), t.recHits(), firstTsos);

}


//
// fit trajectory
//
vector<Trajectory> CosmicMuonSmoother::fit(const TrajectorySeed& seed,
			                  const ConstRecHitContainer& hits, 
				          const TrajectoryStateOnSurface& firstPredTsos) const {

  LogTrace(category_)<< "fit begin (seed, hit, tsos).";

  if ( hits.empty() ) {
    LogTrace(category_)<< "Error: empty hits container.";
    return vector<Trajectory>();
  }

  Trajectory myTraj(seed, alongMomentum);

  TrajectoryStateOnSurface predTsos(firstPredTsos);
  if ( !predTsos.isValid() ) {
    LogTrace(category_)<< "Error: firstTsos invalid.";
    return vector<Trajectory>();
  }
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

    if ((**ihit).isValid() == false && (**ihit).det() == 0) {
      LogTrace(category_)<< "Error: invalid hit.";
      continue;
    }
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

  if ( t.empty() ) {
    LogTrace(category_)<< "Error: smooth: empty trajectory.";
    return vector<Trajectory>();
  }

  Trajectory myTraj(t.seed(), oppositeToMomentum);

  vector<TrajectoryMeasurement> avtm = t.measurements();

  if ( avtm.size() < 2 ) {
    LogTrace(category_)<< "Error: smooth: too little TM. ";
    return vector<Trajectory>();
  }

  TrajectoryStateOnSurface predTsos = avtm.back().forwardPredictedState();
 // predTsos.rescaleError(theErrorRescaling);

  if ( !predTsos.isValid() ) {
    LogTrace(category_)<< "Error: smooth: first TSOS from back invalid. ";
    return vector<Trajectory>();
  }

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
      if ( !combTsos.isValid() ) {
         LogTrace(category_)<< "Error: smooth: combining pred TSOS failed. ";
         return vector<Trajectory>();
      }

      TrajectoryStateOnSurface smooTsos = combiner((*itm).updatedState(), predTsos);

      if ( !smooTsos.isValid() ) {
         LogTrace(category_)<< "Error: smooth: combining smooth TSOS failed. ";
         return vector<Trajectory>();
      }

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
      
      if ( !combTsos.isValid() ) {
         LogTrace(category_)<< "Error: smooth: combining TSOS failed. ";
         return vector<Trajectory>();
      }

      myTraj.push(TrajectoryMeasurement((*itm).forwardPredictedState(),
		     predTsos,
		     combTsos,
		     (*itm).recHit()//,
		     /*(*itm).layer()*/));
    }
  }

  // last smoothed TrajectoryMeasurement is last filtered
  predTsos = propagator()->propagate(currTsos, avtm.front().recHit()->det()->surface());
  
  if ( !predTsos.isValid() ){
    LogTrace(category_)<< "Error: last predict TSOS failed, use original one. ";
 //    return vector<Trajectory>();
      myTraj.push(TrajectoryMeasurement(avtm.front().forwardPredictedState(),
                   avtm.front().recHit()));
  } else  {
    if ( avtm.front().recHit()->isValid() ) {
      //update
      currTsos = theUpdator->update(predTsos, (*avtm.front().recHit()));
      if (currTsos.isValid())
      myTraj.push(TrajectoryMeasurement(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   theEstimator->estimate(predTsos, (*avtm.front().recHit())).second//,
		   /*avtm.front().layer()*/),
	        avtm.front().estimate());
    }
  }
  LogTrace(category_)<< "myTraj foundHits. "<<myTraj.foundHits();

  if (myTraj.foundHits() >= 3)
    return vector<Trajectory>(1, myTraj);
  else {
     LogTrace(category_)<< "Error: smooth: No enough hits in trajctory. ";
     return vector<Trajectory>();
  } 

}

TrajectoryStateOnSurface CosmicMuonSmoother::initialState(const Trajectory& t) const {
  if ( t.empty() ) return TrajectoryStateOnSurface();

  if ( !t.firstMeasurement().updatedState().isValid() || !t.lastMeasurement().updatedState().isValid() )  return TrajectoryStateOnSurface();

  TrajectoryStateOnSurface result;

  bool beamhaloFlag = ( t.firstMeasurement().updatedState().globalMomentum().eta() > 4.0 || t.lastMeasurement().updatedState().globalMomentum().eta() > 4.0 ); 

  if ( !beamhaloFlag ) { //initialState is the top one
     if (t.firstMeasurement().updatedState().globalPosition().y() > t.lastMeasurement().updatedState().globalPosition().y()) {
     result = t.firstMeasurement().updatedState();
     } else {
       result = t.lastMeasurement().updatedState();
     } 
     if (result.globalMomentum().y()> 1.0 ) //top tsos should pointing down
       theUtilities->reverseDirection(result,&*theService->magneticField());
  } else {
     if ( t.firstMeasurement().updatedState().globalPosition().z() * t.lastMeasurement().updatedState().globalPosition().z() >0 ) { //same side
       if ( fabs(t.firstMeasurement().updatedState().globalPosition().z()) > fabs(t.lastMeasurement().updatedState().globalPosition().z()) ) {
         result = t.firstMeasurement().updatedState();
       } else {
         result = t.lastMeasurement().updatedState();
       }
     } else { //different sides

       if ( fabs(t.firstMeasurement().updatedState().globalPosition().eta()) > fabs(t.lastMeasurement().updatedState().globalPosition().eta()) ) {
         result = t.firstMeasurement().updatedState();
       } else {
         result = t.lastMeasurement().updatedState();
       }
     }

  }

  return result;

}

void CosmicMuonSmoother::sortHitsAlongMom(ConstRecHitContainer& hits, const TrajectoryStateOnSurface& tsos) const {

    if (hits.size() < 2) return;
    float dis1 = (hits.front()->globalPosition() - tsos.globalPosition()).mag();
    float dis2 = (hits.back()->globalPosition() - tsos.globalPosition()).mag();

    if ( dis1 > dis2 )
      std::reverse(hits.begin(),hits.end());

    return;
}

