/** \class StandAloneMuonSmoother
 * 
 *  Smooth a trajectory using the standard Kalman Filter smoother.
 *  This class contains the KFTrajectorySmoother and takes care
 *  to update the it whenever the propagator change.
 *
 *  $Date: 2007/02/16 13:31:23 $
 *  $Revision: 1.8 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;


StandAloneMuonSmoother::StandAloneMuonSmoother(const ParameterSet& par, 
					       const MuonServiceProxy* service):theService(service){

  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  
  // The errors of the trajectory state are multiplied by nSigma 
  // to define acceptance of BoundPlane and maximalLocalDisplacement
  theNSigma = par.getParameter<double>("NumberOfSigma"); // default = 3.
  
  // The estimator: makes the decision wheter a measure is good or not
  // it isn't used by the updator which does the real fit. In fact, in principle,
  // a looser request onto the measure set can be requested 
  // (w.r.t. the request on the accept/reject measure in the fit)
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2,theNSigma);
  
  theErrorRescaling = par.getParameter<double>("ErrorRescalingFactor");
  
  thePropagatorName = par.getParameter<string>("Propagator");
  
  theUpdator = new KFUpdator();
  
  // The Kalman smoother
  theSmoother = 0 ;
		
}

StandAloneMuonSmoother::~StandAloneMuonSmoother(){
  if (theEstimator) delete theEstimator;
  if (theUpdator) delete theUpdator;
  if (theSmoother) delete theSmoother;
}

const Propagator* StandAloneMuonSmoother::propagator() const{ 
  return &*theService->propagator(thePropagatorName); 
}

void StandAloneMuonSmoother::renewTheSmoother(){
  if (theService->isTrackingComponentsRecordChanged()){
    if (theSmoother) delete theSmoother;
    theSmoother = new KFTrajectorySmoother(propagator(),
					   updator(),
					   estimator());
  }
  if (!theSmoother)
    theSmoother = new KFTrajectorySmoother(propagator(),
					   updator(),
					   estimator());
  
}

StandAloneMuonSmoother::SmoothingResult StandAloneMuonSmoother::smooth(const Trajectory& inputTrajectory){
  const string metname = "Muon|RecoMuon|StandAloneMuonSmoother";
  
  renewTheSmoother();
  
  vector<Trajectory> trajectoriesSM = smoother()->trajectories(inputTrajectory);
  
  if(!trajectoriesSM.size()){
    LogTrace(metname) << "No Track smoothed!";
    return SmoothingResult(false,inputTrajectory); 
  }
  
  Trajectory smoothed = trajectoriesSM.front(); 
  
  return SmoothingResult(true,smoothed); 
}
