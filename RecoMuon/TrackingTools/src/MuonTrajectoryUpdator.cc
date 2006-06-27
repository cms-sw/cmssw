/** \class MuonTrajectoryUpdator
 *  An updator for the Muon system
 *  This class update a trajectory with a muon chamber measurement.
 *  In spite of the name, it is NOT an updator, but has one.
 *  A muon RecHit is a segment (for DT and CSC) or a "hit" (RPC).
 *  This updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
 *  the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
 *  parameter set, and the propagation direction which is embeded in the propagator set in the c'tor.
 *
 *  $Date: 2006/06/21 17:36:51 $
 *  $Revision: 1.6 $
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

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;;

/// Constructor from Propagator and Parameter set
MuonTrajectoryUpdator::MuonTrajectoryUpdator(Propagator *propagator,
					     const edm::ParameterSet& par):thePropagator(propagator){
  
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();

  // The granularity
  theGranularity = par.getParameter<int>("Granularity");
}

MuonTrajectoryUpdator::MuonTrajectoryUpdator(Propagator *propagator,
					     double chi2, int granularity):theMaxChi2(chi2),
									   theGranularity(granularity),
									   thePropagator(propagator){
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();
}



// FIXME: this c'tor is TMP since it could be dangerous
/// Constructor with Propagator and Parameter set
MuonTrajectoryUpdator::MuonTrajectoryUpdator(const edm::ParameterSet& par){
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);
  
  // The KF updator
  theUpdator= new KFUpdator();
  
  // The granularity
  theGranularity = par.getParameter<int>("Granularity");
}




/// Destructor
MuonTrajectoryUpdator::~MuonTrajectoryUpdator(){
  delete thePropagator;
  delete theEstimator;
  delete theUpdator;
}


// FIXME: check&test!
pair<bool,TrajectoryStateOnSurface> 
MuonTrajectoryUpdator::update(const TrajectoryMeasurement* theMeas, 
			      Trajectory& theTraj){
  
  std::string metname = "Muon|RecoMuon|MuonTrajectoryUpdator";
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
  OwnVector<const TransientTrackingRecHit> recHitsForFit;

  // FIXME: is it right??
  // this are the 4D segment for the CSC/DT and a point for the RPC
  const MuonTransientTrackingRecHit *muonRecHit = 
    dynamic_cast<const MuonTransientTrackingRecHit*> ( theMeas->recHit() ); 

  
  
  switch(theGranularity){
  case 0:
    {
      // Asking for 4D segments for the CSC/DT and a point for the RPC
      recHitsForFit.push_back( muonRecHit->clone() );
      break;
    }
  case 1:
    {
      if (detLayer->module()==dt ) {
	// Asking for 2D segments. theMeas->recHit() returns a 4D segment
	OwnVector<const TransientTrackingRecHit> segments2D = muonRecHit->transientHits();
	// FIXME: this function is not yet available!
	// recHitsForFit.insert(recHitsForFit.end(), segments2D.begin(), segments2D.end());

	// FIXME: remove this as insert will be available
	insert(recHitsForFit,segments2D);

	}

      else if(detLayer->module()==rpc )
	recHitsForFit.push_back( muonRecHit->clone() );

      else if(detLayer->module()==csc) {
	// Asking for 2D points. theMeas->recHit() returns a 4D segment
	OwnVector<const TransientTrackingRecHit> rechit2D = muonRecHit->transientHits();
	// FIXME: this function is not yet available!
	// recHitsForFit.insert(recHitsForFit.end(), rechit2D.begin(), rechit2D.end());

	// FIXME: remove this as insert will be available
	insert(recHitsForFit,rechit2D);
      }
      break;
    }
    
  case 2:
    {
      if (detLayer->module()==dt ) {
	// Asking for 2D segments. theMeas->recHit() returns a 4D segment
	// I have to use OwnVector, since this container must be passed to the
	// KFUpdator, which takes TransientTrackingRecHits...
	OwnVector<const TransientTrackingRecHit> segments2D = muonRecHit->transientHits();
	
	// loop over segment
	for (OwnVector<const TransientTrackingRecHit>::const_iterator segment = segments2D.begin(); 
	     segment != segments2D.end();++segment ){
	  // asking for 1D Rec Hit
	  OwnVector<const TransientTrackingRecHit> rechit1D = (*segment).transientHits();
	  // FIXME: this function is not yet available!
	  // recHitsForFit.insert(recHitsForFit.end(), rechit1D.begin(), rechit1D.end());
	  // FIXME: remove this as insert will be available
	  insert(recHitsForFit,rechit1D);
	}
      }
      else if(detLayer->module()==rpc )
	recHitsForFit.push_back( muonRecHit->clone() );
      
      else if(detLayer->module()==csc) {
	// Asking for 2D points. theMeas->recHit() returns a 4D segment
	OwnVector<const TransientTrackingRecHit> rechit2D = muonRecHit->transientHits();
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
  
  TrajectoryStateOnSurface lastUpdatedTSOS = theMeas->predictedState();
  
  // The OwnVector dataformat doesn't have the revers iterator, so I have to use
  // a trick to move around this.
  OwnVector<const TransientTrackingRecHit>::iterator recHit;
  OwnVector<const TransientTrackingRecHit>::iterator recHitsForFit_begin;
  OwnVector<const TransientTrackingRecHit>::iterator recHitsForFit_end;
  
  // Set the limits according to the propagation direction
  ownVectorLimits(recHitsForFit,recHitsForFit_begin,recHitsForFit_end);
  LogDebug(metname)<<"Own vector size: "<<recHitsForFit.size()<<endl;

  // increment/decrement the iterator according to the propagation direction 
  for(recHit = recHitsForFit_begin; recHit != recHitsForFit_end; incrementIterator(recHit) ) {
    if ((*recHit).isValid() ) {
      // propagate the TSOS onto the rechit plane
      TrajectoryStateOnSurface propagatedTSOS  = propagateState(lastUpdatedTSOS, theMeas, *recHit);
      // // check if the segment have also the zed view (DT)
      // if ( ( newRhit == theMeas.recHit()) ) {
      //   int nptz = newRhit.stereoHit().channels().size();
      //   if ( !nptz ) newRhit = newRhit.monoHit();
      // }
      
      if ( propagatedTSOS.isValid() ) {
        pair<bool,double> thisChi2 = estimator()->estimate(propagatedTSOS, *recHit);

	LogDebug(metname) << "Kalman chi2 " << thisChi2.second;
	
        // The Chi2 cut was already applied in the estimator, which
        // returns 0 if the chi2 is bigger than the cut defined in its
        // constructor
        if ( thisChi2.first ) {
          updated=true;
	  
          LogDebug(metname) << endl 
			    << "     Kalman Start" << endl << endl;
          LogDebug(metname) << "  Meas. Position : " << recHit->globalPosition() << endl
			    << "  Pred. Position : " << propagatedTSOS.globalPosition()
			    << "  Pred Direction : " << propagatedTSOS.globalDirection()<< endl;

          lastUpdatedTSOS = measurementUpdator()->update(propagatedTSOS,*recHit);

          LogDebug(metname) << "  Fit   Position : " << lastUpdatedTSOS.globalPosition()
			    << "  Fit  Direction : " << lastUpdatedTSOS.globalDirection()
			    << endl
			    << "  Fit position radius : " 
			    << lastUpdatedTSOS.globalPosition().perp() << endl
			    << endl << " Filter UPDATED" << endl;
	  
	  muonDumper.dumpTSOS(lastUpdatedTSOS,metname);
	  
	  LogDebug(metname) << "     Kalman End" << endl << endl;	      
	  
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
				      const TransientTrackingRecHit& current) const{

  string tname1 = "MuonTrajectoryUpdator::propagateState::Total";
  TimeMe timer1(tname1);
  const TransientTrackingRecHit *mother = theMeas->recHit();

  // FIXME: check this == !! current and mother could be onto two different unit
  if( current.geographicalId() == mother->geographicalId() )
    return theMeas->predictedState();

  string tname2 = "MuonTrajectoryUpdator::propagateState::Propagation";

  TimeMe timer2(tname2);
  const TrajectoryStateOnSurface  tsos =
    propagator()->propagate(state, current.det()->surface());
  return tsos;

}


void 
MuonTrajectoryUpdator::ownVectorLimits(OwnVector<const TransientTrackingRecHit> &ownvector,
				       OwnVector<const TransientTrackingRecHit>::iterator &recHitsForFit_begin,
				       OwnVector<const TransientTrackingRecHit>::iterator &recHitsForFit_end){
  
  if(propagator()->propagationDirection() == alongMomentum){
    recHitsForFit_begin = ownvector.begin();
    recHitsForFit_end = ownvector.end();
  }
  else if(propagator()->propagationDirection() == oppositeToMomentum){
    recHitsForFit_begin = ownvector.end()-1;
    recHitsForFit_end = ownvector.begin()-1;
  }
  else{
    LogError("MuonTrajectoryUpdator::ownVectorLimits") <<"Wrong propagation direction!!";
  }
}

void 
MuonTrajectoryUpdator::incrementIterator(OwnVector<const TransientTrackingRecHit>::iterator &recHitIterator){

  if(propagator()->propagationDirection() == alongMomentum)
    ++recHitIterator;
  
  else if(propagator()->propagationDirection() == oppositeToMomentum)
    --recHitIterator;
  
  else{
    LogError("MuonTrajectoryUpdator::incrementRecHitIterator") <<"Wrong propagation direction!!";
  }
}

// FIXME: would I a different threatment for the two prop dirrections??
TrajectoryMeasurement MuonTrajectoryUpdator::updateMeasurement(  const TrajectoryStateOnSurface &propagatedTSOS, 
								 const TrajectoryStateOnSurface &lastUpdatedTSOS, 
								 const TransientTrackingRecHit &recHit,
								 const double &chi2, const DetLayer *detLayer, 
								 const TrajectoryMeasurement *initialMeasurement){
   return TrajectoryMeasurement(propagatedTSOS, lastUpdatedTSOS, 
			       recHit.clone(),chi2,detLayer);
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


void MuonTrajectoryUpdator::insert(OwnVector<const TransientTrackingRecHit> & to,
				   OwnVector<const TransientTrackingRecHit> & from){

  for(OwnVector<const TransientTrackingRecHit>::const_iterator it = from.begin();
      it != from.end(); ++it)
    to.push_back(it->clone());
}
