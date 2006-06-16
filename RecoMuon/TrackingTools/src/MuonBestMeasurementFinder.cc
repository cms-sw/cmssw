
/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *  the chi2 cut for the MeasurementEstimator is huge since should not be used.
 *  The aim of this class is to return the "best" measurement according to the
 *  chi2, but without any cut. The decision whether to use or not the
 *  measurement is taken in the caller class.
 *
 *  $Date: 2006/06/12 13:44:03 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// FIXME
using namespace std;

MuonBestMeasurementFinder::MuonBestMeasurementFinder(Propagator* prop):thePropagator(prop){
  theEstimator = new Chi2MeasurementEstimator(100000.);
}

MuonBestMeasurementFinder::~MuonBestMeasurementFinder(){
  delete theEstimator;
}

TrajectoryMeasurement* MuonBestMeasurementFinder::findBestMeasurement(TMContainer& measC){

  std::string metname = "MuonBestMeasurementFinder::findBestMeasurement";

  TimeMe time(metname);

  TMContainer validMeasurements;

  TrajectoryMeasurement* bestMeasurement=0;
  TMIterator measurement;

  // consider only valid TM
  int NumValidMeas=0;
  for ( measurement = measC.begin(); measurement!= measC.end(); ++measurement ) {
    if ((*measurement).recHit()->isValid()) {
      ++NumValidMeas;
      bestMeasurement = &(*measurement);
      validMeasurements.push_back(*measurement);
    }
  }

  // If we have just one (or zero) valid meas, return it at once 
  // (or return null measurement)
  if(NumValidMeas<=1) {
    LogDebug(metname) << "MuonBestMeasurement: just " << NumValidMeas
		      << " valid measurement ";
    // FIXME
    cout << "MuonBestMeasurement: just " << NumValidMeas 
	 << " valid measurement"<<endl;

    return bestMeasurement;
  }

  double minChi2PerNDoF=1.E6;

  // if there are more than one valid measurement, then sort them.
  for ( measurement = validMeasurements.begin(); measurement!= validMeasurements.end(); measurement++ ) {

    const TransientTrackingRecHit *measRH= (*measurement).recHit();
    
    unsigned int npts=0;
    double thisChi2 = 0.;
    
    // ask for the segments
    TransientTrackingRecHit::RecHitContainer rhits_list = measRH->transientHits();
    
    // loop over them
    for (TransientTrackingRecHit::RecHitContainer::const_iterator rhit = rhits_list.begin(); 
	 rhit!= rhits_list.end(); rhit++ ) {
      if ((*rhit).isValid() ) {
	npts+=(*rhit).dimension();
	  
	TrajectoryStateOnSurface predState;

	// Double FIXME
	if (!( (*rhit).geographicalId() == (*measRH).geographicalId() ) )
	  predState = propagator()->propagate(*(*measurement).predictedState().freeState(),
					      (*rhit).det()->surface()); 
	
	else predState = (*measurement).predictedState();  
	  
	if ( predState.isValid() ) { 
	  std::pair<bool,double> sing_chi2 = estimator()->estimate( predState, *rhit);
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
      bestMeasurement = &(*measurement);
    }
    
  }
  
  return bestMeasurement;
}

// OLD ORCA algo. Reported for timing comparison pourpose
// Will be removed after the comparison!
TrajectoryMeasurement* MuonBestMeasurementFinder::findBestMeasurement_OLD(TMContainer& measC){

  std::string metname = "MuonBestMeasurementFinder::findBestMeasurement_OLD";

  TimeMe time(metname);

  TrajectoryMeasurement* theMeas=0;
  TMIterator meas;

  // consider only valid TM
  int NumValidMeas=0;
  for ( meas = measC.begin(); meas!= measC.end(); meas++ ) {
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

    const TransientTrackingRecHit *measRH= (*meas).recHit();

    if ( measRH->isValid() ) {
      unsigned int npts=0;
      double thisChi2 = 0.;

      // ask for the segments
    TransientTrackingRecHit::RecHitContainer rhits_list = measRH->transientHits();
      
      // loop over them
      for (TransientTrackingRecHit::RecHitContainer::const_iterator rhit = rhits_list.begin(); 
           rhit!= rhits_list.end(); rhit++ ) {
        if ((*rhit).isValid() ) {
          npts+=(*rhit).dimension();
	  
          TrajectoryStateOnSurface predState;

	  // FIXME
   	  // was (! *rhit == measRH)
	  if (!( (*rhit).geographicalId() == (*measRH).geographicalId() ) ) {
	    predState = propagator()->propagate(*(*meas).predictedState().freeState(),
						(*rhit).det()->surface()); 
	       // FIXME was (*rhit).det().detUnits()[0]->specificSurface()
	  }
	  else {
	    predState = (*meas).predictedState();  
	  }
	  
          if ( predState.isValid() ) { 
            std::pair<bool,double> sing_chi2 = estimator()->estimate( predState, *rhit);
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


