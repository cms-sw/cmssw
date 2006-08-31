/** \class MuonTrajectoryUpdator
 *  An updator for the Muon system
 *  This class update a trajectory with a muon chamber measurement.
 *  In spite of the name, it is NOT an updator, but has one.
 *  A muon RecHit is a segment (for DT and CSC) or a "hit" (RPC).
 *  This updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
 *  the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
 *  parameter set, and the propagation direction which is embeded in the propagator set in the c'tor.
 *
 *  $Date: 2006/08/22 15:17:58 $
 *  $Revision: 1.16 $
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
MuonTrajectoryUpdator::MuonTrajectoryUpdator( PropagationDirection fitDirection,
					      const edm::ParameterSet& par): theFitDirection(fitDirection){
  
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();

  // The granularity
  theGranularity = par.getParameter<int>("Granularity");
}

MuonTrajectoryUpdator::MuonTrajectoryUpdator( PropagationDirection fitDirection,
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
MuonTrajectoryUpdator::update(const TrajectoryMeasurement* theMeas,
			      Trajectory& theTraj,
			      const Propagator *propagator){
  
  const std::string metname = "Muon|RecoMuon|MuonTrajectoryUpdator";
  TimeMe t(metname);
  MuonPatternRecoDumper muonDumper;

  // Status of the updating
  bool updated=false;
  
  // FIXM put a warning
  if(!theMeas) return pair<bool,TrajectoryStateOnSurface>(updated,TrajectoryStateOnSurface() );

  // measurement layer
  const DetLayer* detLayer=theMeas->layer();

  // Must be an own vector.
  // The KFUpdator takes TransientTrackingRecHits as arg.
  TransientTrackingRecHit::ConstRecHitContainer recHitsForFit;
 
  // this are the 4D segment for the CSC/DT and a point for the RPC 
  TransientTrackingRecHit::ConstRecHitPointer muonRecHit=  ((theMeas->recHit()));
 
  //   LogDebug(metname)<<"The granulaity is "<<theGranularity;
  //   LogDebug(metname)<<"The detLayer type is ";
 
  //   if(detLayer->module() == dt)
  //     LogDebug(metname)<<"DT"<<endl;
  //   else if(detLayer->module() == csc)
  //     LogDebug(metname)<<"CSC"<<endl;
  //   else if(detLayer->module() == rpc)
  //     LogDebug(metname)<<"RPC"<<endl;
  //   else 
  //     LogDebug(metname)<<"I don't know!"<<endl;
 
  switch(theGranularity){
  case 0:
    {
      // Asking for 4D segments for the CSC/DT and a point for the RPC
      //     recHitsForFit.push_back( muonRecHit->clone() );
      recHitsForFit.push_back( muonRecHit );
      break;
    }
  case 1:
    {
      if (detLayer->subDetector()==GeomDetEnumerators::DT ) {
	// Asking for 2D segments. theMeas->recHit() returns a 4D segment
	TransientTrackingRecHit::ConstRecHitContainer segments2D = muonRecHit->transientHits();
	// FIXME: this function is not yet available!
	// recHitsForFit.insert(recHitsForFit.end(), segments2D.begin(), segments2D.end());
	
	// FIXME: remove this as insert will be available
	insert(recHitsForFit,segments2D);
      }
      
      else if(detLayer->subDetector()==GeomDetEnumerators::RPCBarrel || detLayer->subDetector()==GeomDetEnumerators::RPCEndcap)
	recHitsForFit.push_back( muonRecHit);


      
      else if(detLayer->subDetector()==GeomDetEnumerators::CSC) {
	// Asking for 2D points. theMeas->recHit() returns a 4D segment
	TransientTrackingRecHit::ConstRecHitContainer rechit2D = muonRecHit->transientHits();
	// FIXME: this function is not yet available!
	// recHitsForFit.insert(recHitsForFit.end(), rechit2D.begin(), rechit2D.end());

	// FIXME: remove this as insert will be available
	insert(recHitsForFit,rechit2D);
      }
      break;
    }
    
  case 2:
    {
      if (detLayer->subDetector()==GeomDetEnumerators::DT ) {
	// Asking for 2D segments. theMeas->recHit() returns a 4D segment
	// I have to use OwnVector, since this container must be passed to the
	// KFUpdator, which takes TransientTrackingRecHits...
	TransientTrackingRecHit::ConstRecHitContainer segments2D = muonRecHit->transientHits();
	
	// loop over segment
	for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator segment = segments2D.begin(); 
	     segment != segments2D.end();++segment ){

	  // asking for 1D Rec Hit
	  TransientTrackingRecHit::ConstRecHitContainer rechit1D = (**segment).transientHits();
	  
	  // FIXME: this function is not yet available!
	  // recHitsForFit.insert(recHitsForFit.end(), rechit1D.begin(), rechit1D.end());

	  // FIXME: remove this as insert will be available
	  insert(recHitsForFit,rechit1D);	  
	}
      }
      else if(detLayer->subDetector()==GeomDetEnumerators::RPCBarrel || detLayer->subDetector()==GeomDetEnumerators::RPCEndcap)
	recHitsForFit.push_back(muonRecHit  );
      
      else if(detLayer->subDetector()==GeomDetEnumerators::CSC) {
	// Asking for 2D points. theMeas->recHit() returns a 4D segment
	TransientTrackingRecHit::ConstRecHitContainer rechit2D = (*muonRecHit).transientHits();

	// FIXME: this function is not yet available!
	// recHitsForFit.insert(recHitsForFit.end(), rechit2D.begin(), rechit2D.end());
	
	// FIXME: remove this as insert will be available
	insert(recHitsForFit,rechit2D);
      }
      
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
  sort(recHitsForFit,detLayer,propagator->propagationDirection());
  
  TrajectoryStateOnSurface lastUpdatedTSOS = theMeas->predictedState();
  
  LogDebug(metname)<<"Own vector size: "<<recHitsForFit.size()<<endl;
 
  TransientTrackingRecHit::ConstRecHitContainer::iterator recHit;
  for(recHit = recHitsForFit.begin(); recHit != recHitsForFit.end(); ++recHit ) {
    if ((*recHit)->isValid() ) {

      // propagate the TSOS onto the rechit plane
      TrajectoryStateOnSurface propagatedTSOS  = propagateState(lastUpdatedTSOS, theMeas, 
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
									theMeas);
	  // FIXME: check!
	  theTraj.push(updatedMeasurement, thisChi2.second);	  
	}
      }
    }
  }
  return pair<bool,TrajectoryStateOnSurface>(updated,lastUpdatedTSOS);
}



TrajectoryStateOnSurface 
MuonTrajectoryUpdator::propagateState(const TrajectoryStateOnSurface& state,
				      const TrajectoryMeasurement* theMeas, 
				      const TransientTrackingRecHit::ConstRecHitPointer  & current,
				      const Propagator *propagator) const{

  string tname1 = "MuonTrajectoryUpdator::propagateState::Total";
  TimeMe timer1(tname1);
  const TransientTrackingRecHit::ConstRecHitPointer mother = theMeas->recHit();

  if( current->geographicalId() == mother->geographicalId() )
    return theMeas->predictedState();
  
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


void MuonTrajectoryUpdator::insert (TransientTrackingRecHit::ConstRecHitContainer & to,
				    TransientTrackingRecHit::ConstRecHitContainer & from){


  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator it = from.begin();
      it != from.end(); ++it)
    to.push_back(*it);
}


void MuonTrajectoryUpdator::sort(TransientTrackingRecHit::ConstRecHitContainer& recHitsForFit, const DetLayer* detLayer,PropagationDirection propDirection){

  if(detLayer->subDetector()==GeomDetEnumerators::DT){
    if(propDirection == alongMomentum)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), RadiusComparatorInOut() );
    else if(propDirection == oppositeToMomentum)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(),RadiusComparatorOutIn() );

    // If the propagation direction is "undefined" then use the global fit direction to sort the RecHits
    else if(propDirection == anyDirection){
      if(theFitDirection == alongMomentum)
	stable_sort(recHitsForFit.begin(),recHitsForFit.end(), RadiusComparatorInOut() );
      else if(theFitDirection == oppositeToMomentum)
	stable_sort(recHitsForFit.begin(),recHitsForFit.end(),RadiusComparatorOutIn() );
      else
	LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong fit direction!!";
    }      
    
    else{
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
    }
  }
  else if(detLayer->subDetector()==GeomDetEnumerators::CSC){
    if(propDirection == alongMomentum)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorInOut() );
    else if(propDirection == oppositeToMomentum)
      stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorOutIn() );

    // If the propagation direction is "undefined" then use the global fit direction to sort the RecHits
    else if(propDirection == anyDirection){
      if(theFitDirection == alongMomentum)
	stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorInOut() );
      else if(theFitDirection == oppositeToMomentum)
	stable_sort(recHitsForFit.begin(),recHitsForFit.end(), ZedComparatorOutIn() );
      else
	LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong fit direction!!";
    }
    
    else{
      LogError("Muon|RecoMuon|MuonTrajectoryUpdator") <<"MuonTrajectoryUpdator::sort: Wrong propagation direction!!";
    }
  }
}
