#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

using namespace std;

TransientInitialStateEstimator::TransientInitialStateEstimator( const edm::EventSetup& es,
								const edm::ParameterSet& conf)
{
  thePropagatorAlongName    = conf.getParameter<std::string>("propagatorAlongTISE");   
  thePropagatorOppositeName = conf.getParameter<std::string>("propagatorOppositeTISE");   
  theNumberMeasurementsForFit = conf.getParameter<int32_t>("numberMeasurementsForFit");   


  // let's avoid breaking compatibility now
  es.get<TrackingComponentsRecord>().get(thePropagatorAlongName,thePropagatorAlong);
  es.get<TrackingComponentsRecord>().get(thePropagatorOppositeName,thePropagatorOpposite);
}

void TransientInitialStateEstimator::setEventSetup( const edm::EventSetup& es ) {
  es.get<TrackingComponentsRecord>().get(thePropagatorAlongName,thePropagatorAlong);
  es.get<TrackingComponentsRecord>().get(thePropagatorOppositeName,thePropagatorOpposite);
}

std::pair<TrajectoryStateOnSurface, const GeomDet*> 
TransientInitialStateEstimator::innerState( const Trajectory& traj, bool doBackFit) const
{
  if (!doBackFit && traj.firstMeasurement().forwardPredictedState().isValid()){
    LogDebug("TransientInitialStateEstimator")
      <<"a backward fit will not be done. assuming that the state on first measurement is OK";
    TSOS firstStateFromForward = traj.firstMeasurement().forwardPredictedState();
    firstStateFromForward.rescaleError(100.);    
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>( std::move(firstStateFromForward), 
								traj.firstMeasurement().recHit()->det());
  }
  if (!doBackFit){
    LogDebug("TransientInitialStateEstimator")
      <<"told to not do a back fit, but the forward state of the first measurement is not valid. doing a back fit.";
  }

  int nMeas = traj.measurements().size();
  int lastFitted = theNumberMeasurementsForFit >=0 ? theNumberMeasurementsForFit : nMeas; 
  if (nMeas-1 < lastFitted) lastFitted = nMeas-1;

  auto const & measvec = traj.measurements();
  TransientTrackingRecHit::ConstRecHitContainer firstHits;

  bool foundLast = false;
  int actualLast = -99;

  for (int i=lastFitted; i >= 0; i--) {
    if(measvec[i].recHit()->det()){
      if(!foundLast){
	actualLast = i; 
	foundLast = true;
      }
      firstHits.push_back( measvec[i].recHit());
    }
  }
  TSOS startingState = measvec[actualLast].updatedState();
  startingState.rescaleError(100.);

  // avoid cloning...
  KFUpdator const aKFUpdator;
  Chi2MeasurementEstimator const aChi2MeasurementEstimator( 100., 3);
  KFTrajectoryFitter backFitter( thePropagatorAlong.product(),
				 &aKFUpdator,
				 &aChi2MeasurementEstimator,
				 firstHits.size());

  PropagationDirection backFitDirection = traj.direction() == alongMomentum ? oppositeToMomentum: alongMomentum;

  // only direction matters in this contest
  TrajectorySeed fakeSeed (PTrajectoryStateOnDet() , 
		           edm::OwnVector<TrackingRecHit>(),
		           backFitDirection);

  Trajectory && fitres = backFitter.fitOne( fakeSeed, firstHits, startingState, traj.nLoops()>0 ?  TrajectoryFitter::looper : TrajectoryFitter::standard);
  
  LogDebug("TransientInitialStateEstimator")
    <<"using a backward fit of :"<<firstHits.size()<<" hits, starting from:\n"<<startingState
    <<" to get the estimate of the initial state of the track.";

  if (!fitres.isValid()) {
        LogDebug("TransientInitialStateEstimator")
	  << "FitTester: first hits fit failed!";
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement const & firstMeas = fitres.lastMeasurement();

  // magnetic field can be different!
  TSOS firstState(firstMeas.updatedState().localParameters(),
		  firstMeas.updatedState().localError(),
  		  firstMeas.updatedState().surface(),
  		  thePropagatorAlong->magneticField());
  
 
  // TSOS & firstState = const_cast<TSOS&>(firstMeas.updatedState());

  // this fails in case of zero field? (for sure for beamhalo reconstruction)
  // assert(thePropagatorAlong->magneticField()==firstState.magneticField());

  firstState.rescaleError(100.);

  LogDebug("TransientInitialStateEstimator")
    <<"the initial state is found to be:\n:"<<firstState
    <<"\n it's field pointer is: "<<firstState.magneticField()
    <<"\n the pointer from the state of the back fit was: "<<firstMeas.updatedState().magneticField();


  return std::pair<TrajectoryStateOnSurface, const GeomDet*>( std::move(firstState), 
							      firstMeas.recHit()->det());
}

