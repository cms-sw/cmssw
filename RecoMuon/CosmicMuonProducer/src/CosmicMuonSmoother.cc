/** \file CosmicMuonSmoother
 *
 *  The class refit muon trajectories based on MuonTrackReFitter
 *  But 
 *  (1) check direction to make sure it's downward for cosmic
 *  (2) propagate by virtual intermediate planes if failed to ensure propagation
 *      within cylinders
 *
 *
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
  thePropagatorAlongName = par.getParameter<string>("PropagatorAlong");
  thePropagatorOppositeName = par.getParameter<string>("PropagatorOpposite");
  theErrorRescaling = par.getParameter<double>("RescalingFactor");

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
Trajectory CosmicMuonSmoother::trajectory(const Trajectory& t) const {
   std::vector<Trajectory> && fitted = fit(t);
   if (fitted.empty()) return  Trajectory();
   std::vector<Trajectory> && smoothed = smooth(fitted);
   return  smoothed.empty() ? Trajectory() : smoothed.front(); 
}

//
// fit and smooth trajectory 
//
std::vector<Trajectory> CosmicMuonSmoother::trajectories(const TrajectorySeed& seed,
	                                           const ConstRecHitContainer& hits, 
	                                           const TrajectoryStateOnSurface& firstPredTsos) const {

  if ( hits.empty() ||!firstPredTsos.isValid() ) return vector<Trajectory>();

  LogTrace(category_)<< "trajectory begin (seed hits tsos)";

  TrajectoryStateOnSurface firstTsos = firstPredTsos;
  firstTsos.rescaleError(theErrorRescaling);

  LogTrace(category_)<< "first TSOS: "<<firstTsos;

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);
  LogTrace(category_)<< "fitted: "<<fitted.size();
  vector<Trajectory> smoothed = smooth(fitted);
  LogTrace(category_)<< "smoothed: "<<smoothed.size();

  return smoothed;

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
  LogTrace(category_)<<"hit front" <<hits.front()->globalPosition()<< " hit back" 
    <<hits.back()->globalPosition();

  sortHitsAlongMom(hits, firstTsos);

  LogTrace(category_)<<"after sorting hit front" <<hits.front()->globalPosition()<< " hit back" 
    <<hits.back()->globalPosition();

  return fit(t.seed(), hits, firstTsos);

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
  LogTrace(category_)<< "first pred TSOS: "<<predTsos;

  if ( !predTsos.isValid() ) {
    LogTrace(category_)<< "Error: firstTsos invalid.";
    return vector<Trajectory>();
  }
  TrajectoryStateOnSurface currTsos;

  if ( hits.front()->isValid() ) {

    // FIXME  FIXME  CLONE !!!
   //  TrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);
   auto preciseHit = hits.front();
 
    LogTrace(category_)<<"first hit is at det "<< hits.front()->det()->surface().position();

    currTsos = theUpdator->update(predTsos, *preciseHit);
    if (!currTsos.isValid()){
      edm::LogWarning(category_)
	<<"an updated state is not valid. killing the trajectory.";
      return vector<Trajectory>();
    }
    myTraj.push(TrajectoryMeasurement(predTsos, currTsos, hits.front(),
                theEstimator->estimate(predTsos, *hits.front()).second));
    if ( currTsos.isValid() )  LogTrace(category_)<< "first curr TSOS: "<<currTsos;

  } else {

    currTsos = predTsos;
    myTraj.push(TrajectoryMeasurement(predTsos, hits.front()));
  }
  //const TransientTrackingRecHit& firsthit = *hits.front();

  for ( ConstRecHitContainer::const_iterator ihit = hits.begin() + 1; 
        ihit != hits.end(); ++ihit ) {

    if ( !(**ihit).isValid() ) {
      LogTrace(category_)<< "Error: invalid hit.";
      continue;
    }
   if (currTsos.isValid())  {
     LogTrace(category_)<<"current pos "<<currTsos.globalPosition()
                       <<"mom "<<currTsos.globalMomentum();
    } else {
      LogTrace(category_)<<"current state invalid";
    }

    predTsos = propagatorAlong()->propagate(currTsos, (**ihit).det()->surface());
    LogTrace(category_)<<"predicted state propagate directly "<<predTsos.isValid();

    if ( !predTsos.isValid() ) {
      LogTrace(category_)<<"step-propagating from "<<currTsos.globalPosition() <<" to position: "<<(*ihit)->globalPosition();
      predTsos = theUtilities->stepPropagate(currTsos, (*ihit), *propagatorAlong());
    }
    if ( !predTsos.isValid() && (fabs(theService->magneticField()->inTesla(GlobalPoint(0,0,0)).z()) < 0.01) && (theService->propagator("StraightLinePropagator").isValid() ) ) {
      LogTrace(category_)<<"straight-line propagating from "<<currTsos.globalPosition() <<" to position: "<<(*ihit)->globalPosition();
      predTsos = theService->propagator("StraightLinePropagator")->propagate(currTsos, (**ihit).det()->surface());
    }
    if (predTsos.isValid())  {
      LogTrace(category_)<<"predicted pos "<<predTsos.globalPosition()
                         <<"mom "<<predTsos.globalMomentum();
    } else {
      LogTrace(category_)<<"predicted state invalid";
    }
    if ( !predTsos.isValid() ) {
      LogTrace(category_)<< "Error: predTsos is still invalid forward fit.";
//      return vector<Trajectory>();
      continue;
    } else if ( (**ihit).isValid() ) {
          // FIXME  FIXME  CLONE !!!
      // update  (FIXME!)
      // TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);
      auto preciseHit = *ihit;
 
      if ( !preciseHit->isValid() ) {
        currTsos = predTsos;
        myTraj.push(TrajectoryMeasurement(predTsos, *ihit));
      } else {
        currTsos = theUpdator->update(predTsos, *preciseHit);
	if (!currTsos.isValid()){
	  edm::LogWarning(category_)
	    <<"an updated state is not valid. killing the trajectory.";
	  return vector<Trajectory>();
	}
        myTraj.push(TrajectoryMeasurement(predTsos, currTsos, preciseHit,
                       theEstimator->estimate(predTsos, *preciseHit).second));
      }
    } else {
      currTsos = predTsos;
      myTraj.push(TrajectoryMeasurement(predTsos, *ihit));
    }

  }

  std::vector<TrajectoryMeasurement> mytms = myTraj.measurements();
  LogTrace(category_)<<"fit result "<<mytms.size();
  for (std::vector<TrajectoryMeasurement>::const_iterator itm = mytms.begin();
       itm != mytms.end(); ++itm ) {
       LogTrace(category_)<<"updated pos "<<itm->updatedState().globalPosition()
                       <<"mom "<<itm->updatedState().globalMomentum();
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
  predTsos.rescaleError(theErrorRescaling);

  if ( !predTsos.isValid() ) {
    LogTrace(category_)<< "Error: smooth: first TSOS from back invalid. ";
    return vector<Trajectory>();
  }

  TrajectoryStateOnSurface currTsos;

  // first smoothed TrajectoryMeasurement is last fitted
  if ( avtm.back().recHit()->isValid() ) {
    currTsos = theUpdator->update(predTsos, (*avtm.back().recHit()));
    if (!currTsos.isValid()){
      edm::LogWarning(category_)
	<<"an updated state is not valid. killing the trajectory.";
      return vector<Trajectory>();
    }
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

   if (currTsos.isValid())  {
     LogTrace(category_)<<"current pos "<<currTsos.globalPosition()
                       <<"mom "<<currTsos.globalMomentum();
    } else {
      LogTrace(category_)<<"current state invalid";
    }

    predTsos = propagatorOpposite()->propagate(currTsos,(*itm).recHit()->det()->surface());

    if ( !predTsos.isValid() ) {
      LogTrace(category_)<<"step-propagating from "<<currTsos.globalPosition() <<" to position: "<<(*itm).recHit()->globalPosition();
      predTsos = theUtilities->stepPropagate(currTsos, (*itm).recHit(), *propagatorOpposite());
    }
   if (predTsos.isValid())  {
      LogTrace(category_)<<"predicted pos "<<predTsos.globalPosition()
                       <<"mom "<<predTsos.globalMomentum();
    } else {
      LogTrace(category_)<<"predicted state invalid";
    }

    if ( !predTsos.isValid() ) {
      LogTrace(category_)<< "Error: predTsos is still invalid backward smooth.";
      return vector<Trajectory>();
    //  continue;
    } else if ( (*itm).recHit()->isValid() ) {
      //update
      currTsos = theUpdator->update(predTsos, (*(*itm).recHit()));
      if (!currTsos.isValid()){
	edm::LogWarning(category_)
	  <<"an updated state is not valid. killing the trajectory.";
	return vector<Trajectory>();
      }
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
  predTsos = propagatorOpposite()->propagate(currTsos, avtm.front().recHit()->det()->surface());
  
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

