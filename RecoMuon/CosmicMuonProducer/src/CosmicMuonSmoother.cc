/** \file CosmicMuonSmoother
 *
 *  The class refit muon trajectories based on MuonTrackReFitter
 *  But 
 *  (1) check direction to make sure it's downward for cosmic
 *  (2) propagate by virtual intermediate planes if failed to ensure propagation
 *      within cylinders
 *
 *
 *  $Date: $
 *  $Revision: $
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
  theEstimator   = new Chi2MeasurementEstimator(200.0);
  thePropagatorName = par.getParameter<string>("Propagator");

}

//
// destructor
//
CosmicMuonSmoother::~CosmicMuonSmoother() {

  if ( theUpdator ) delete theUpdator;
  if ( theEstimator ) delete theEstimator;

}


//
// fit and smooth trajectory
//
vector<Trajectory> CosmicMuonSmoother::trajectories(const Trajectory& t) const {
   vector<Trajectory> fitted = fit(t);
   return smooth(fitted);

}

void CosmicMuonSmoother::reverseDirection(TrajectoryStateOnSurface& tsos) const {

   GlobalTrajectoryParameters gtp(tsos.globalPosition(),
                                  -tsos.globalMomentum(),
                                  -tsos.charge(),
                                  &*theService->magneticField()  );
   TrajectoryStateOnSurface newTsos(gtp, tsos.cartesianError(), tsos.surface()); 
   tsos = newTsos;
   return;

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

  if (firstTsos.globalMomentum().y()>0 && firstTsos.globalMomentum().eta()< 4.0 ) 
     reverseDirection(firstTsos);

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

       predTsos = stepPropagate(currTsos, (*ihit));

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

  if ( predTsos.globalMomentum().y() > 0 && firstTsos.globalMomentum().eta()< 4.0 )  reverseDirection(predTsos);

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
       predTsos = stepPropagate(currTsos, (*itm).recHit());
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


void CosmicMuonSmoother::print(const MuonTransientTrackingRecHit::ConstMuonRecHitContainer& hits) const {

    const std::string metname = "Muon|RecoMuon|CosmicMuonSmoother";

    for (ConstMuonRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug(metname) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug(metname)
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector();
  }

}

TrajectoryStateOnSurface CosmicMuonSmoother::stepPropagate(const TrajectoryStateOnSurface& tsos,
                                              const ConstRecHitPointer& hit) const {

  const std::string metname = "Muon|RecoMuon|CosmicMuonSmoother";

  GlobalPoint start = tsos.globalPosition();
  GlobalPoint dest = hit->globalPosition();
  GlobalVector StepVector = dest - start;
  GlobalVector UnitStepVector = StepVector.unit();
  GlobalPoint GP =start;
  TrajectoryStateOnSurface result(tsos);
  float totalDis = StepVector.mag();
  LogDebug(metname)<<"stepPropagate: propagate from: "<<start<<" to "<<dest;
  LogDebug(metname)<<"stepPropagate: their distance: "<<totalDis;

  int steps = 3; // need to optimize

  float oneStep = totalDis/steps;
  Basic3DVector<float> Basic3DV(StepVector.x(),StepVector.y(),StepVector.z());
  for ( int istep = 0 ; istep < steps - 1 ; istep++) {
        GP += oneStep*UnitStepVector;
        Surface::PositionType pos(GP.x(),GP.y(),GP.z());
        LogDebug(metname)<<"stepPropagate: a middle plane: "<<pos;
        Surface::RotationType rot( Basic3DV , float(0));
        PlaneBuilder::ReturnType SteppingPlane = PlaneBuilder().plane(pos,rot);
        TrajectoryStateOnSurface predTsos = propagator()->propagate( result, *SteppingPlane);
        if (predTsos.isValid()) {
            result=predTsos;
            LogDebug(metname)<<"result "<< result.globalPosition();
          }
 }

  TrajectoryStateOnSurface predTsos = propagator()->propagate( result, hit->det()->surface());
  if (predTsos.isValid()) result=predTsos;

  return result;
}


void CosmicMuonSmoother::print(const TransientTrackingRecHit::ConstRecHitContainer& hits) const {

    const std::string metname = "Muon|RecoMuon|CosmicMuonSmoother";

    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogDebug(metname) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogDebug(metname)
    << "pos"<<pos
    << "radius "<<pos.perp()
    << "  dim " << (*ir)->dimension()
    << "  det " << (*ir)->det()->geographicalId().det()
    << "  sub det " << (*ir)->det()->subDetector();
  }

}
