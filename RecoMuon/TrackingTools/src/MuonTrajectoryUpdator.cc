/** \class MuonTrajectoryUpdator
 *  An updator for the Muon system
 *  This class update a trajectory with a muon chamber measurement.
 *  In spite of the name, it is NOT an updator, but has one.
 *  A muon RecHit is a segment (for DT and CSC) or a "hit" (RPC).
 *  This updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
 *  the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
 *  parameter set, and the propagation direction which is embeded in the propagator set in the c'tor.
 *
 *  $Date: 2006/09/04 17:13:12 $
 *  $Revision: 1.20 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */


#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

using namespace edm;
using namespace std;

/// Constructor from Propagator and Parameter set
MuonTrajectoryUpdator::MuonTrajectoryUpdator(const edm::ParameterSet& par,
					     recoMuon::FitDirection fitDirection): theFitDirection(fitDirection){
  
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();

  // The granularity
  theGranularity = par.getParameter<int>("Granularity");

  // Rescale the error of the first state?
  theRescaleErrorFlag =  par.getParameter<bool>("RescaleError");

  if(theRescaleErrorFlag)
    // The rescale factor
    theRescaleFactor =  par.getParameter<double>("RescaleErrorFactor");
  
  // Flag needed for the rescaling
  theFirstTSOSFlag = true;
}

MuonTrajectoryUpdator::MuonTrajectoryUpdator( recoMuon::FitDirection fitDirection,
					      double chi2, int granularity): theMaxChi2(chi2),
									     theGranularity(granularity),
									     theFitDirection(fitDirection){
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();
}

/// Destructor
MuonTrajectoryUpdator::~MuonTrajectoryUpdator(){
  delete theEstimator;
  delete theUpdator;
}

pair<bool,TrajectoryStateOnSurface> 
MuonTrajectoryUpdator::update(const TrajectoryMeasurement* measurement,
			      Trajectory& trajectory,
			      const Propagator *propagator){
  
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryUpdator";
  TimeMe t(metname);
  MuonPatternRecoDumper muonDumper;

  // Status of the updating
  bool updated=false;
  
  if(!measurement) return pair<bool,TrajectoryStateOnSurface>(updated,TrajectoryStateOnSurface() );

  // measurement layer
  const DetLayer* detLayer=measurement->layer();

  // The KFUpdator takes TransientTrackingRecHits as arg.
  TransientTrackingRecHit::ConstRecHitContainer recHitsForFit;
 
  // this are the 4D segment for the CSC/DT and a point for the RPC 
  TransientTrackingRecHit::ConstRecHitPointer muonRecHit = measurement->recHit();
 
  switch(theGranularity){
  case 0:
    {
      // Asking for 4D segments for the CSC/DT and a point for the RPC
      recHitsForFit.push_back( muonRecHit );
      break;
    }
  case 1:
    {
      if (detLayer->subDetector()==GeomDetEnumerators::DT ||
	  detLayer->subDetector()==GeomDetEnumerators::CSC) 
	// measurement->recHit() returns a 4D segment, then
	// DT case: asking for 2D segments.
	// CSC case: asking for 2D points.
	recHitsForFit = muonRecHit->transientHits();
      
      else if(detLayer->subDetector()==GeomDetEnumerators::RPCBarrel || 
	      detLayer->subDetector()==GeomDetEnumerators::RPCEndcap)
	recHitsForFit.push_back( muonRecHit);   
      
      break;
    }
    
  case 2:
    {
      if (detLayer->subDetector()==GeomDetEnumerators::DT ) {

	// Asking for 2D segments. measurement->recHit() returns a 4D segment
	TransientTrackingRecHit::ConstRecHitContainer segments2D = muonRecHit->transientHits();
	
	// loop over segment
	for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator segment = segments2D.begin(); 
	     segment != segments2D.end();++segment ){

	  // asking for 1D Rec Hit
	  TransientTrackingRecHit::ConstRecHitContainer rechit1D = (**segment).transientHits();
	  
	  // load them into the recHitsForFit container
	  copy(rechit1D.begin(),rechit1D.end(),back_inserter(recHitsForFit));
	}
      }

      else if(detLayer->subDetector()==GeomDetEnumerators::RPCBarrel || 
	      detLayer->subDetector()==GeomDetEnumerators::RPCEndcap)
	recHitsForFit.push_back(muonRecHit);
      
      else if(detLayer->subDetector()==GeomDetEnumerators::CSC)	
	// Asking for 2D points. measurement->recHit() returns a 4D segment
	recHitsForFit = (*muonRecHit).transientHits();
      
      break;
    }
    
  default:
    {
      throw cms::Exception(metname) <<"Wrong granularity chosen!"
				    <<"it will be set to 0";
      break;
    }
  }

  // sort the container in agreement with the porpagation direction
  sort(recHitsForFit,detLayer);
  
  TrajectoryStateOnSurface lastUpdatedTSOS = measurement->predictedState();
  
  LogDebug(metname)<<"Number of rechits for the fit: "<<recHitsForFit.size()<<endl;
 
  TransientTrackingRecHit::ConstRecHitContainer::iterator recHit;
  for(recHit = recHitsForFit.begin(); recHit != recHitsForFit.end(); ++recHit ) {
    if ((*recHit)->isValid() ) {

      // propagate the TSOS onto the rechit plane
      TrajectoryStateOnSurface propagatedTSOS  = propagateState(lastUpdatedTSOS, measurement, 
								*recHit, propagator);
      
      if ( propagatedTSOS.isValid() ) {
        pair<bool,double> thisChi2 = estimator()->estimate(propagatedTSOS, *((*recHit).get()));

	LogDebug(metname) << "Estimation for Kalman Fit. Chi2: " << thisChi2.second;
	
        // The Chi2 cut was already applied in the estimator, which
        // returns 0 if the chi2 is bigger than the cut defined in its
        // constructor
        if ( thisChi2.first ) {
          updated=true;
	  
          LogDebug(metname) << endl 
			    << "     Kalman Start" << "\n" << "\n";
          LogDebug(metname) << "  Meas. Position : " << (**recHit).globalPosition() << "\n"
			    << "  Pred. Position : " << propagatedTSOS.globalPosition()
			    << "  Pred Direction : " << propagatedTSOS.globalDirection()<< endl;

	  if(theRescaleErrorFlag && theFirstTSOSFlag){
	    propagatedTSOS.rescaleError(theRescaleFactor);
	    theFirstTSOSFlag = false;
	  }

          lastUpdatedTSOS = measurementUpdator()->update(propagatedTSOS,*((*recHit).get()));

          LogDebug(metname) << "  Fit   Position : " << lastUpdatedTSOS.globalPosition()
			    << "  Fit  Direction : " << lastUpdatedTSOS.globalDirection()
			    << "\n"
			    << "  Fit position radius : " 
			    << lastUpdatedTSOS.globalPosition().perp()
			    << "filter updated" << endl;
	  
	  LogDebug(metname) << muonDumper.dumpTSOS(lastUpdatedTSOS);
	  
	  LogDebug(metname) << "\n\n     Kalman End" << "\n" << "\n";	      
	  
	  TrajectoryMeasurement updatedMeasurement = updateMeasurement( propagatedTSOS, lastUpdatedTSOS, 
									*recHit,thisChi2.second,detLayer, 
									measurement);
	  // FIXME: check!
	  trajectory.push(updatedMeasurement, thisChi2.second);	  
	}
      }
    }
  }
  return pair<bool,TrajectoryStateOnSurface>(updated,lastUpdatedTSOS);
}

TrajectoryStateOnSurface 
MuonTrajectoryUpdator::propagateState(const TrajectoryStateOnSurface& state,
				      const TrajectoryMeasurement* measurement, 
				      const TransientTrackingRecHit::ConstRecHitPointer  & current,
				      const Propagator *propagator) const{

  string tname1 = "MuonTrajectoryUpdator::propagateState::Total";
  TimeMe timer1(tname1);
  const TransientTrackingRecHit::ConstRecHitPointer mother = measurement->recHit();

  if( current->geographicalId() == mother->geographicalId() )
    return measurement->predictedState();
  
  string tname2 = "MuonTrajectoryUpdator::propagateState::Propagation";

  TimeMe timer2(tname2);
  const TrajectoryStateOnSurface  tsos =
    propagator->propagate(state, current->det()->surface());
  return tsos;

}

// FIXME: would I a different threatment for the two prop dirrections??
TrajectoryMeasurement MuonTrajectoryUpdator::updateMeasurement(  const TrajectoryStateOnSurface &propagatedTSOS, 
								 const TrajectoryStateOnSurface &lastUpdatedTSOS, 
								 const TransientTrackingRecHit::ConstRecHitPointer &recHit,
								 const double &chi2, const DetLayer *detLayer, 
								 const TrajectoryMeasurement *initialMeasurement){
  return TrajectoryMeasurement(propagatedTSOS, lastUpdatedTSOS, 
			       recHit,chi2,detLayer);

  //   // FIXME: put a better check! One could fit in first out-in and then in - out 
  //   if(propagator()->propagationDirection() == alongMomentum) 
  //     return TrajectoryMeasurement(propagatedTSOS, lastUpdatedTSOS, 
  // 				 recHit,thisChi2.second,detLayer);
  
  //   // FIXME: Check this carefully!!
  //   else if(propagator()->propagationDirection() == oppositeToMomentum)
  //     return TrajectoryMeasurement(initialMeasurement->forwardPredictedState(),
  // 				 propagatedTSOS, lastUpdatedTSOS, 
  // 				 recHit,thisChi2.second,detLayer);
  //   else{
  //     LogError("MuonTrajectoryUpdator::updateMeasurement") <<"Wrong propagation direction!!";
  //   }
}


void MuonTrajectoryUpdator::sort(TransientTrackingRecHit::ConstRecHitContainer& recHitsForFit, 
				 const DetLayer* detLayer){
  
  if(detLayer->subDetector()==GeomDetEnumerators::DT){
    if(fitDirection() == recoMuon::insideOut)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), RadiusComparatorInOut() );
    else if(fitDirection() == recoMuon::outsideIn)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(),RadiusComparatorOutIn() ); 
    else
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
  }

  else if(detLayer->subDetector()==GeomDetEnumerators::CSC){
    if(fitDirection() == recoMuon::insideOut)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorInOut() );
    else if(fitDirection() == recoMuon::outsideIn)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorOutIn() );  
    else
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
  }
}
