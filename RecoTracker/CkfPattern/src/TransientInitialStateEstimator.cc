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
								const edm::ParameterSet& conf):
  thePropagatorWatcher([this](TrackingComponentsRecord const& iRecord) {
      {
	edm::ESHandle<Propagator> propHandle;
	iRecord.get(thePropagatorAlongName,propHandle);
	thePropagatorAlong.reset(propHandle->clone());
      }
      {
	edm::ESHandle<Propagator> propHandle;
	iRecord.get(thePropagatorOppositeName,propHandle);
	thePropagatorOpposite.reset(propHandle->clone());
      }
    })
{
  thePropagatorAlongName    = conf.getParameter<std::string>("propagatorAlongTISE");   
  thePropagatorOppositeName = conf.getParameter<std::string>("propagatorOppositeTISE");   
  theNumberMeasurementsForFit = conf.getParameter<int32_t>("numberMeasurementsForFit");   


  // let's avoid breaking compatibility now
  thePropagatorWatcher.check(es);
}

void TransientInitialStateEstimator::setEventSetup( const edm::EventSetup& es ) {
  thePropagatorWatcher.check(es);
}

std::pair<TrajectoryStateOnSurface, const GeomDet*> 
TransientInitialStateEstimator::innerState( const Trajectory& traj, bool doBackFit) const
{
  if (!doBackFit && traj.firstMeasurement().forwardPredictedState().isValid()){
    LogDebug("TransientInitialStateEstimator")
      <<"a backward fit will not be done. assuming that the state on first measurement is OK";
    TSOS firstStateFromForward = traj.firstMeasurement().forwardPredictedState();
    firstStateFromForward.rescaleError(100.);    
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>( firstStateFromForward, 
								traj.firstMeasurement().recHit()->det());
  }
  if (!doBackFit){
    LogDebug("TransientInitialStateEstimator")
      <<"told to not do a back fit, but the forward state of the first measurement is not valid. doing a back fit.";
  }

  int nMeas = traj.measurements().size();
  int lastFitted = theNumberMeasurementsForFit >=0 ? theNumberMeasurementsForFit : nMeas; 
  if (nMeas-1 < lastFitted) lastFitted = nMeas-1;

  std::vector<TrajectoryMeasurement> measvec = traj.measurements();
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
  KFTrajectoryFitter backFitter( thePropagatorAlong.get(),
				 &aKFUpdator,
				 &aChi2MeasurementEstimator,
				 firstHits.size());

  PropagationDirection backFitDirection = traj.direction() == alongMomentum ? oppositeToMomentum: alongMomentum;

  // only direction matters in this contest
  TrajectorySeed fakeSeed = TrajectorySeed(PTrajectoryStateOnDet() , 
					   edm::OwnVector<TrackingRecHit>(),
					   backFitDirection);

  vector<Trajectory> fitres = backFitter.fit( fakeSeed, firstHits, startingState);
  
  LogDebug("TransientInitialStateEstimator")
    <<"using a backward fit of :"<<firstHits.size()<<" hits, starting from:\n"<<startingState
    <<" to get the estimate of the initial state of the track.";

  if (fitres.size() != 1) {
        LogDebug("TransientInitialStateEstimator")
	  << "FitTester: first hits fit failed!";
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement firstMeas = fitres[0].lastMeasurement();
  TSOS firstState(firstMeas.updatedState().localParameters(),
		  firstMeas.updatedState().localError(),
  		  firstMeas.updatedState().surface(),
  		  thePropagatorAlong->magneticField());
  // I couldn't do: 
  //TSOS firstState = firstMeas.updatedState();
  // why????


  firstState.rescaleError(100.);

  LogDebug("TransientInitialStateEstimator")
    <<"the initial state is found to be:\n:"<<firstState
    <<"\n it's field pointer is: "<<firstState.magneticField()
    <<"\n the pointer from the state of the back fit was: "<<firstMeas.updatedState().magneticField();


  return std::pair<TrajectoryStateOnSurface, const GeomDet*>( firstState, 
							      firstMeas.recHit()->det());
}

