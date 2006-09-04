/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *  the chi2 cut for the MeasurementEstimator is huge since should not be used.
 *  The aim of this class is to return the "best" measurement according to the
 *  chi2, but without any cut. The decision whether to use or not the
 *  measurement is taken in the caller class.
 *  The evaluation is made (in hard-code way) with the granularity = 1. Where
 *  the granularity is the one defined in the MuonTrajectoyUpdatorClass.
 *
 *  $Date: 2006/08/31 18:24:18 $
 *  $Revision: 1.10 $
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
#include "Utilities/Timing/interface/TimingReport.h"

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

  TimeMe time(metname);

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
    LogDebug(metname) << "MuonBestMeasurement: just " << NumValidMeas
		      << " valid measurement ";
    return bestMeasurement;
  }

  TMIterator measurement;
  double minChi2PerNDoF=1.E6;

  // if there are more than one valid measurement, then sort them.
  for ( measurement = validMeasurements.begin(); measurement!= validMeasurements.end(); measurement++ ) {

    TransientTrackingRecHit::ConstRecHitPointer muonRecHit = (*measurement)->recHit();
    
    unsigned int npts=0;
    double thisChi2 = 0.;
    
    // ask for the 2D-segments/2D-rechit
    TransientTrackingRecHit::ConstRecHitContainer rhits_list = muonRecHit->transientHits();

    LogDebug(metname)<<"Number of rechits in the measurement rechit: "<<rhits_list.size()<<endl;
    
    // loop over them
    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit = rhits_list.begin(); 
	 rhit!= rhits_list.end(); rhit++ ) {
      if ((*rhit)->isValid() ) {
	LogDebug(metname)<<"Rechit dimension: "<<(*rhit)->dimension()<<endl;
	npts+=(*rhit)->dimension();
	  
	TrajectoryStateOnSurface predState;

	if (!( (*rhit)->geographicalId() == (*muonRecHit).geographicalId() ) ){
	  predState = propagator->propagate(*(*measurement)->predictedState().freeState(),
					      (*rhit)->det()->surface()); 
	}
	else predState = (*measurement)->predictedState();  
	  
	if ( predState.isValid() ) { 
	  std::pair<bool,double> sing_chi2 = estimator()->estimate( predState, *((*rhit).get()));
	  thisChi2 += sing_chi2.second ;
	  LogDebug(metname) << " sing_chi2: "
			    << sing_chi2.second;
	}
      }
    }
    double chi2PerNDoF = thisChi2/npts;
    LogDebug(metname) << " The eeasurement has a chi2/npts " << chi2PerNDoF << " with dof = " << npts
		      << " \n Till now the best chi2 is " << minChi2PerNDoF;
    
    if ( chi2PerNDoF && chi2PerNDoF<minChi2PerNDoF ) {
      minChi2PerNDoF = chi2PerNDoF;	
      bestMeasurement = *measurement;
    }
    
  }
  LogDebug(metname)<<"The final best chi2 is "<<minChi2PerNDoF<<endl;
  return bestMeasurement;
}

// TO BE fixxed with the new transient Hits() method.
// OLD ORCA algo. Reported for timing comparison pourpose
// Will be removed after the comparison!
TrajectoryMeasurement* 
MuonBestMeasurementFinder::findBestMeasurement_OLD(std::vector<TrajectoryMeasurement>& measC,
						   const Propagator* propagator){

  const std::string metname = "Muon|RecoMuon|MuonBestMeasurementFinder";

  TimeMe time(metname);

  TrajectoryMeasurement* theMeas=0;
  vector<TrajectoryMeasurement>::iterator meas;

  // consider only valid TM
  int NumValidMeas=0;
  for ( meas = measC.begin(); 
	meas!= measC.end(); meas++ ) {
    if ((*meas).recHit()->isValid()) {
      NumValidMeas++;
      theMeas = &(*meas);
    }
  }

  // If we have just one (or zero) valid meas, return it at once 
  // (or return null meas)
  if(NumValidMeas<=1) {
    LogDebug(metname) << "MuonBestMeasurement: just " << NumValidMeas
		      << " valid meas ";
    return theMeas;
  }

  double minChi2PerNDoF=1.E6;
  // if there are more than one valid meas, then sort them.
  for ( meas = measC.begin(); meas!= measC.end(); meas++ ) {

    TransientTrackingRecHit::ConstRecHitPointer measRH= (*meas).recHit();
    
    if ( measRH->isValid() ) {
      unsigned int npts=0;
      double thisChi2 = 0.;

      // ask for the segments
      TransientTrackingRecHit::ConstRecHitContainer rhits_list = measRH->transientHits();
      
      // loop over them
      for (TransientTrackingRecHit::RecHitContainer::const_iterator rhit = rhits_list.begin(); 
           rhit!= rhits_list.end(); rhit++ ) {
        if ((*rhit)->isValid() ) {
          npts+=(*rhit)->dimension();
	  
          TrajectoryStateOnSurface predState;

	  // FIXME
   	  // was (! *rhit == measRH)
	  if (!( (*rhit)->geographicalId() ==
		 (*measRH).geographicalId() ) ) {
	    predState = propagator->propagate(*(*meas).predictedState().freeState(),
					      (*rhit)->det()->surface()); 
	       // FIXME was (*rhit).det().detUnits()[0]->specificSurface()
	  }
	  else {
	    predState = (*meas).predictedState();  
	  }
	  
          if ( predState.isValid() ) { 
            std::pair<bool,double> sing_chi2 = estimator()->estimate( predState, *((*rhit).get()));
            thisChi2 += sing_chi2.second ;
            LogDebug(metname) << " sing_chi2: "
			      << sing_chi2.second;
          }
        }
      }
      double chi2PerNDoF = thisChi2/npts;
      LogDebug(metname) << " --> chi2/npts " << chi2PerNDoF << "/" << npts
			<< " best chi2=" << minChi2PerNDoF;

      if ( chi2PerNDoF && chi2PerNDoF<minChi2PerNDoF ) {
        minChi2PerNDoF = chi2PerNDoF;	
        theMeas = &(*meas);
      }
    }
    else {
      LogDebug(metname) << "MuonBestMeasurement Invalid Meas";
    }
  }

  return theMeas;
 
}


