/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *  the chi2 cut for the MeasurementEstimator is huge since should not be used.
 *  The aim of this class is to return the "best" measurement according to the
 *  chi2, but without any cut. The decision whether to use or not the
 *  measurement is taken in the caller class.
 *  The evaluation is made (in hard-code way) with the granularity = 1. Where
 *  the granularity is the one defined in the MuonTrajectoyUpdatorClass.
 *
 *  $Date: 2009/10/31 02:05:11 $
 *  $Revision: 1.16 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace std;

MuonBestMeasurementFinder::MuonBestMeasurementFinder(){
  
  theEstimator = new Chi2MeasurementEstimator(100000.);
}

MuonBestMeasurementFinder::~MuonBestMeasurementFinder(){
  delete theEstimator;
}

TrajectoryMeasurement* 
MuonBestMeasurementFinder::findBestMeasurement(std::vector<TrajectoryMeasurement>& measC,
					       const Propagator* propagator){
  
  const std::string metname = "Muon|RecoMuon|MuonBestMeasurementFinder";

  TMContainer validMeasurements;

  TrajectoryMeasurement* bestMeasurement=0;

  // consider only valid TM
  int NumValidMeas=0;
  for ( vector<TrajectoryMeasurement>::iterator measurement = measC.begin(); 
	measurement!= measC.end(); ++measurement ) {
    if ((*measurement).recHit()->isValid()) {
      ++NumValidMeas;
      bestMeasurement = &(*measurement);
      validMeasurements.push_back( &(*measurement) );
    }
  }

  // If we have just one (or zero) valid meas, return it at once 
  // (or return null measurement)
  if(NumValidMeas<=1) {
    LogTrace(metname) << "MuonBestMeasurement: just " << NumValidMeas
		      << " valid measurement ";
    return bestMeasurement;
  }

  TMIterator measurement;
  double minChi2PerNDoF=1.E6;

  // if there are more than one valid measurement, then sort them.
  for ( measurement = validMeasurements.begin(); measurement!= validMeasurements.end(); measurement++ ) {

    TransientTrackingRecHit::ConstRecHitPointer muonRecHit = (*measurement)->recHit();
    
    // FIXME!! FIXME !! FIXME !!
    pair<double,int> chi2Info = lookAtSubRecHits(*measurement, propagator);

    double chi2PerNDoF = chi2Info.first/chi2Info.second;
    LogTrace(metname) << " The measurement has a chi2/npts " << chi2PerNDoF << " with dof = " << chi2Info.second
		      << " \n Till now the best chi2 is " << minChi2PerNDoF;
    
    if ( chi2PerNDoF && chi2PerNDoF<minChi2PerNDoF ) {
      minChi2PerNDoF = chi2PerNDoF;	
      bestMeasurement = *measurement;
    }
    
  }
  LogTrace(metname)<<"The final best chi2 is "<<minChi2PerNDoF<<endl;
  return bestMeasurement;
}


pair<double,int> MuonBestMeasurementFinder::lookAtSubRecHits(TrajectoryMeasurement* measurement,
							     const Propagator* propagator){
  
  const std::string metname = "Muon|RecoMuon|MuonBestMeasurementFinder";

  unsigned int npts=0;
  // unused  double thisChi2 = 0.;

  TransientTrackingRecHit::ConstRecHitPointer muonRecHit = measurement->recHit();
  TrajectoryStateOnSurface predState = measurement->predictedState();                          // temporarily introduced by DT
  std::pair<bool, double> sing_chi2 = estimator()->estimate( predState, *(muonRecHit.get()));  // temporarily introduced by DT
  npts = 1;                                                                                    // temporarily introduced by DT
  std::pair<double, int> result = pair<double, int>(sing_chi2.second, npts);                   // temporarily introduced by DT
  
//   // ask for the 2D-segments/2D-rechit                                                      // temporarily excluded by DT
//   TransientTrackingRecHit::ConstRecHitContainer rhits_list = muonRecHit->transientHits();
  
//   LogTrace(metname)<<"Number of rechits in the measurement rechit: "<<rhits_list.size()<<endl;
  
//   // loop over them
//   for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit = rhits_list.begin();
//        rhit!= rhits_list.end(); ++rhit)
//     if ((*rhit)->isValid() ) {
//       LogTrace(metname)<<"Rechit dimension: "<<(*rhit)->dimension()<<endl;
//       npts+=(*rhit)->dimension();
      
//       TrajectoryStateOnSurface predState;
      
//       if (!( (*rhit)->geographicalId() == muonRecHit->geographicalId() ) ){
// 	predState = propagator->propagate(*measurement->predictedState().freeState(),
// 					  (*rhit)->det()->surface()); 
//       }
//       else predState = measurement->predictedState();  
      
//       if ( predState.isValid() ) { 
// 	std::pair<bool,double> sing_chi2 = estimator()->estimate( predState, *((*rhit).get()));
// 	thisChi2 += sing_chi2.second ;
// 	LogTrace(metname) << " single incremental chi2: " << sing_chi2.second;
//       }
//     }
  
//   pair<double,int> result = pair<double,int>(thisChi2,npts);
  return result;
}
