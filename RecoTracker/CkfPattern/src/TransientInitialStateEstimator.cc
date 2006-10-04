#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"


TransientInitialStateEstimator::TransientInitialStateEstimator( const edm::EventSetup& es,
								const edm::ParameterSet& conf)
{
  std::string propagatorAlongName    = conf.getParameter<std::string>("propagatorAlongTISE");   
  std::string propagatorOppositeName = conf.getParameter<std::string>("propagatorOppositeTISE");   

  es.get<TrackingComponentsRecord>().get(propagatorAlongName,theForwardPropagator);
  es.get<TrackingComponentsRecord>().get(propagatorOppositeName,theReversePropagator);
}


std::pair<TrajectoryStateOnSurface, const GeomDet*> 
TransientInitialStateEstimator::innerState( const Trajectory& traj) const
{
  int lastFitted = 4;
  int nhits = traj.foundHits();
  if (nhits < lastFitted+1) lastFitted = nhits-1;

  std::vector<TrajectoryMeasurement> measvec = traj.measurements();
  TransientTrackingRecHit::ConstRecHitContainer firstHits;

  bool foundLast = false;
  int actualLast = -99;
  for (int i=lastFitted; i >= 0; i--) {
    if(measvec[i].recHit()->isValid()){
      if(!foundLast){
	actualLast = i; 
	foundLast = true;
      }
      firstHits.push_back( measvec[i].recHit());
    }
  }
  TSOS unscaledState = measvec[actualLast].updatedState();
  AlgebraicSymMatrix C(5,1);
  // C *= 100.;

  TSOS startingState( unscaledState.localParameters(), LocalTrajectoryError(C),
		      unscaledState.surface(),
		      theReversePropagator->magneticField());

  // cout << endl << "FitTester starts with state " << startingState << endl;

  KFTrajectoryFitter backFitter( *theReversePropagator, KFUpdator(), 
				 Chi2MeasurementEstimator( 100., 3));

  vector<Trajectory> fitres = backFitter.fit( traj.seed(), firstHits, startingState);

  if (fitres.size() != 1) {
    // cout << "FitTester: first hits fit failed!" << endl;
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement firstMeas = fitres[0].lastMeasurement();
  TSOS firstState = firstMeas.updatedState();

  //  cout << "FitTester: Fitted first state " << firstState << endl;
  //cout << "FitTester: chi2 = " << fitres[0].chiSquared() << endl;

  TSOS initialState( firstState.localParameters(), LocalTrajectoryError(C),
		     firstState.surface(),
		     theReversePropagator->magneticField());

  return std::pair<TrajectoryStateOnSurface, const GeomDet*>( initialState, 
							      firstMeas.recHit()->det());
}

