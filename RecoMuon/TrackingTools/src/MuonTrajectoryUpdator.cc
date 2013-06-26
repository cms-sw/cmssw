/** \class MuonTrajectoryUpdator
 *  An updator for the Muon system
 *  This class update a trajectory with a muon chamber measurement.
 *  In spite of the name, it is NOT an updator, but has one.
 *  A muon RecHit is a segment (for DT and CSC) or a "hit" (RPC).
 *  This updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
 *  the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
 *  parameter set, and the propagation direction which is embeded in the propagator set in the c'tor.
 *
 *  $Date: 2010/11/18 12:02:16 $
 *  $Revision: 1.38 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */


#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBreaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

using namespace edm;
using namespace std;

/// Constructor from Propagator and Parameter set
MuonTrajectoryUpdator::MuonTrajectoryUpdator(const edm::ParameterSet& par,
					     NavigationDirection fitDirection): theFitDirection(fitDirection){
  
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
  
  // Use the invalid hits?
  useInvalidHits =  par.getParameter<bool>("UseInvalidHits");

  // Flag needed for the rescaling
  theFirstTSOSFlag = true;

  // Exlude RPC from the fit?
   theRPCExFlag = par.getParameter<bool>("ExcludeRPCFromFit");
}

MuonTrajectoryUpdator::MuonTrajectoryUpdator( NavigationDirection fitDirection,
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

void MuonTrajectoryUpdator::makeFirstTime(){
  theFirstTSOSFlag = true;
}


pair<bool,TrajectoryStateOnSurface> 
MuonTrajectoryUpdator::update(const TrajectoryMeasurement* measurement,
			      Trajectory& trajectory,
			      const Propagator *propagator){
  
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryUpdator";

  MuonPatternRecoDumper muonDumper;

  // Status of the updating
  bool updated=false;
  
  if(!measurement) return pair<bool,TrajectoryStateOnSurface>(updated,TrajectoryStateOnSurface() );

  // measurement layer
  const DetLayer* detLayer=measurement->layer();

  // these are the 4D segment for the CSC/DT and a point for the RPC 
  TransientTrackingRecHit::ConstRecHitPointer muonRecHit = measurement->recHit();
 
  // The KFUpdator takes TransientTrackingRecHits as arg.
  TransientTrackingRecHit::ConstRecHitContainer recHitsForFit =
  MuonTransientTrackingRecHitBreaker::breakInSubRecHits(muonRecHit,theGranularity);

  // sort the container in agreement with the porpagation direction
  sort(recHitsForFit,detLayer);
  
  TrajectoryStateOnSurface lastUpdatedTSOS = measurement->predictedState();
  
  LogTrace(metname)<<"Number of rechits for the fit: "<<recHitsForFit.size()<<endl;
 
  TransientTrackingRecHit::ConstRecHitContainer::iterator recHit;
  for(recHit = recHitsForFit.begin(); recHit != recHitsForFit.end(); ++recHit ) {
    if ((*recHit)->isValid() ) {

      // propagate the TSOS onto the rechit plane
      TrajectoryStateOnSurface propagatedTSOS  = propagateState(lastUpdatedTSOS, measurement, 
								*recHit, propagator);
      
      if ( propagatedTSOS.isValid() ) {
        pair<bool,double> thisChi2 = estimator()->estimate(propagatedTSOS, *((*recHit).get()));

	LogTrace(metname) << "Estimation for Kalman Fit. Chi2: " << thisChi2.second;

	// Is an RPC hit? Prepare the variable to possibly exluding it from the fit
	bool wantIncludeThisHit = true;
	if (theRPCExFlag && 
	    (*recHit)->geographicalId().det() == DetId::Muon &&
	    (*recHit)->geographicalId().subdetId() == MuonSubdetId::RPC){
	  wantIncludeThisHit = false;	
	  LogTrace(metname) << "This is an RPC hit and the present configuration is such that it will be excluded from the fit";
	}

	
        // The Chi2 cut was already applied in the estimator, which
        // returns 0 if the chi2 is bigger than the cut defined in its
        // constructor
	if (thisChi2.first) {
          updated=true;
	  if (wantIncludeThisHit) { // This split is a trick to have the RPC hits counted as updatable (in used chamber counting), while are not actually included in the fit when the proper obtion is activated.

          LogTrace(metname) << endl 
			    << "     Kalman Start" << "\n" << "\n";
          LogTrace(metname) << "  Meas. Position : " << (**recHit).globalPosition() << "\n"
			    << "  Pred. Position : " << propagatedTSOS.globalPosition()
			    << "  Pred Direction : " << propagatedTSOS.globalDirection()<< endl;

	  if(theRescaleErrorFlag && theFirstTSOSFlag){
	    propagatedTSOS.rescaleError(theRescaleFactor);
	    theFirstTSOSFlag = false;
	  }

          lastUpdatedTSOS = measurementUpdator()->update(propagatedTSOS,*((*recHit).get()));

          LogTrace(metname) << "  Fit   Position : " << lastUpdatedTSOS.globalPosition()
			    << "  Fit  Direction : " << lastUpdatedTSOS.globalDirection()
			    << "\n"
			    << "  Fit position radius : " 
			    << lastUpdatedTSOS.globalPosition().perp()
			    << "filter updated" << endl;
	  
	  LogTrace(metname) << muonDumper.dumpTSOS(lastUpdatedTSOS);
	  
	  LogTrace(metname) << "\n\n     Kalman End" << "\n" << "\n";	      
	  
	  TrajectoryMeasurement updatedMeasurement = updateMeasurement( propagatedTSOS, lastUpdatedTSOS, 
									*recHit, thisChi2.second, detLayer, 
									measurement);
	  // FIXME: check!
	  trajectory.push(updatedMeasurement, thisChi2.second);	
	  }
	  else {
	    LogTrace(metname) << "  Compatible RecHit with good chi2 but made with RPC when it was decided to not include it in the fit"
			      << "  --> trajectory NOT updated, invalid RecHit added." << endl;
	      
	    MuonTransientTrackingRecHit::MuonRecHitPointer invalidRhPtr = MuonTransientTrackingRecHit::specificBuild( (*recHit)->det(), (*recHit)->hit() );
	    invalidRhPtr->invalidateHit();
	    TrajectoryMeasurement invalidRhMeasurement(propagatedTSOS, propagatedTSOS, invalidRhPtr.get(), thisChi2.second, detLayer);
	    trajectory.push(invalidRhMeasurement, thisChi2.second);	  	    
	  }
	}
	else {
          if(useInvalidHits) {
            LogTrace(metname) << "  Compatible RecHit with too large chi2"
			    << "  --> trajectory NOT updated, invalid RecHit added." << endl;

	    MuonTransientTrackingRecHit::MuonRecHitPointer invalidRhPtr = MuonTransientTrackingRecHit::specificBuild( (*recHit)->det(), (*recHit)->hit() );
	    invalidRhPtr->invalidateHit();
	    TrajectoryMeasurement invalidRhMeasurement(propagatedTSOS, propagatedTSOS, invalidRhPtr.get(), thisChi2.second, detLayer);
	    trajectory.push(invalidRhMeasurement, thisChi2.second);	  
          }
	}
      }
    }
  }
  recHitsForFit.clear();
  return pair<bool,TrajectoryStateOnSurface>(updated,lastUpdatedTSOS);
}

TrajectoryStateOnSurface 
MuonTrajectoryUpdator::propagateState(const TrajectoryStateOnSurface& state,
				      const TrajectoryMeasurement* measurement, 
				      const TransientTrackingRecHit::ConstRecHitPointer  & current,
				      const Propagator *propagator) const{

  const TransientTrackingRecHit::ConstRecHitPointer mother = measurement->recHit();

  if( current->geographicalId() == mother->geographicalId() )
    return measurement->predictedState();
  
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
    if(fitDirection() == insideOut)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), RadiusComparatorInOut() );
    else if(fitDirection() == outsideIn)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(),RadiusComparatorOutIn() ); 
    else
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
  }

  else if(detLayer->subDetector()==GeomDetEnumerators::CSC){
    if(fitDirection() == insideOut)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorInOut() );
    else if(fitDirection() == outsideIn)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorOutIn() );  
    else
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
  }
}
