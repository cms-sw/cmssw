/**  \class MuonTrackReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: 2006/08/09 21:56:09 $
 *   $Revision: 1.1 $
 *
 *   \author   N. Neumeister            Purdue University
 *   \author   C. Liu                   Purdue University
 **/

#include "RecoMuon/TrackingTools/interface/MuonTrackReFitter.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

//
// constructor
//
MuonTrackReFitter::MuonTrackReFitter(const ParameterSet& par) {

  theErrorRescaling = 100.0;
 
  theUpdator     = new KFUpdator;
  theEstimator   = new Chi2MeasurementEstimator(200.0);

  std::string inPropagatorAlongMom = par.getParameter<string>("InPropagatorAlongMom");
  std::string outPropagatorAlongMom = par.getParameter<string>("OutPropagatorAlongMom");
  std::string inPropagatorOppositeToMom = par.getParameter<string>("InPropagatorOppositeToMom");
  std::string outPropagatorOppositeToMom = par.getParameter<string>("OutPropagatorOppositeToMom");

}


//
// destructor
//
MuonTrackReFitter::~MuonTrackReFitter() {

  if ( thePropagator2 ) delete thePropagator2;
  if ( thePropagator1 ) delete thePropagator1;
  if ( theUpdator ) delete theUpdator;
  if ( theEstimator ) delete theEstimator;

}


//
// initialize propagators
//
void MuonTrackReFitter::setES(const edm::EventSetup& iSetup) {

  ESHandle<Propagator> eshPropagator1;
  iSetup.get<TrackingComponentsRecord>().get(theOutPropagatorAlongMom, eshPropagator1);

  ESHandle<Propagator> eshPropagator2;
  iSetup.get<TrackingComponentsRecord>().get(theInPropagatorAlongMom, eshPropagator2);

  ESHandle<Propagator> eshPropagator3;
  iSetup.get<TrackingComponentsRecord>().get(theOutPropagatorOppositeToMom, eshPropagator3);
 
  ESHandle<Propagator> eshPropagator4;
  iSetup.get<TrackingComponentsRecord>().get(theInPropagatorOppositeToMom, eshPropagator4);

  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

  thePropagator1 = new SmartPropagator(*eshPropagator2,*eshPropagator1, &*theMagField);
  thePropagator2 = new SmartPropagator(*eshPropagator4,*eshPropagator3, &*theMagField,oppositeToMomentum);

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
  firstTsos.rescaleError(3.);

  vector<Trajectory> fitted = fit(seed, hits, firstTsos);

  return smooth(fitted);

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

  //Trajectory myTraj(seed, thePropagator1->propagationDirection());
  Trajectory myTraj(seed, alongMomentum);

  TSOS predTsos(firstPredTsos);
  if ( !predTsos.isValid() ) return vector<Trajectory>();

  TSOS currTsos;

  if ( hits.front()->isValid() ) {
    // update
    currTsos = theUpdator->update(predTsos, *hits.front());
    myTraj.push(TM(predTsos, currTsos, hits.front(),
                   theEstimator->estimate(predTsos, *hits.front()).second));
  } else {
    currTsos = predTsos;
    myTraj.push(TM(predTsos, hits.front()));
  }

  for ( ConstRecHitContainer::const_iterator ihit = hits.begin() + 1; 
        ihit != hits.end(); ++ihit ) {

    predTsos = thePropagator1->propagate(currTsos, (**ihit).det()->surface());

    if ( !predTsos.isValid() ) {
      //return vector<Trajectory>();
    } else if ( (**ihit).isValid() ) {
      // update
      currTsos = theUpdator->update(predTsos, **ihit);
      myTraj.push(TM(predTsos, currTsos, *ihit,
                     theEstimator->estimate(predTsos, **ihit).second));
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
//
//
vector<Trajectory> MuonTrackReFitter::smooth(const Trajectory& t) const {

  if ( t.empty() ) return vector<Trajectory>();

  //Trajectory myTraj(t.seed(), thePropagator2->propagationDirection());
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
