#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


BaseCkfTrajectoryBuilder::
BaseCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
			 const TrajectoryStateUpdator*         updator,
			 const Propagator*                     propagatorAlong,
			 const Propagator*                     propagatorOpposite,
			 const Chi2MeasurementEstimatorBase*   estimator,
			 const TransientTrackingRecHitBuilder* recHitBuilder,
			 const MeasurementTracker*             measurementTracker):
  theUpdator(updator),
  thePropagatorAlong(propagatorAlong),thePropagatorOpposite(propagatorOpposite),
  theEstimator(estimator),theTTRHBuilder(recHitBuilder),
  theMeasurementTracker(measurementTracker),
  theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
  theForwardPropagator(0),theBackwardPropagator(0),
  theMaxLostHit(conf.getParameter<int>("maxLostHit")),
  theMaxConsecLostHit(conf.getParameter<int>("maxConsecLostHit")),
  theMinimumNumberOfHits(conf.getParameter<int>("minimumNumberOfHits")),
  theChargeSignificance(conf.getParameter<double>("chargeSignificance")),
  theMinPtCondition(new MinPtTrajectoryFilter(conf.getParameter<double>("ptCut"))),
  theMaxHitsCondition(new MaxHitsTrajectoryFilter(conf.getParameter<int>("maxNumberOfHits")))
{}
 
BaseCkfTrajectoryBuilder::~BaseCkfTrajectoryBuilder(){
  delete theMinPtCondition;
  delete theMaxHitsCondition;
  delete theLayerMeasurements;
}


void
BaseCkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed,  std::vector<TrajectoryMeasurement> & result) const
{
  TrajectoryStateTransform tsTransform;

  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = 
      theMeasurementTracker->geomTracker()->idToDet( ihit->geographicalId());

    const DetLayer* hitLayer = 
      theMeasurementTracker->geometricSearchTracker()->detLayer(ihit->geographicalId());

    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    if (ihit == hitRange.second - 1) {
      // the seed trajectory state should correspond to this hit
      PTrajectoryStateOnDet pState( seed.startingState());
      const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet( DetId(pState.detId()));
      if (&gdet->surface() != &hitGeomDet->surface()) {
	edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
	return; // FIXME: should throw exception
      }

      TSOS updatedState = tsTransform.transientState( pState, &(gdet->surface()), 
						      theForwardPropagator->magneticField());
      result.push_back(TM( invalidState, updatedState, recHit, 0, hitLayer));
    }
    else {
      PTrajectoryStateOnDet pState( seed.startingState());

      TSOS outerState = tsTransform.transientState(pState,
						   &((theMeasurementTracker->geomTracker()->idToDet(
										     (hitRange.second - 1)->geographicalId()))->surface()),  
						   theForwardPropagator->magneticField());
      TSOS innerState   = theBackwardPropagator->propagate(outerState,hitGeomDet->surface());
      if(innerState.isValid()) {
	TSOS innerUpdated = theUpdator->update(innerState,*recHit);
	result.push_back(TM( invalidState, innerUpdated, recHit, 0, hitLayer));
      }
    }
  }

  // method for debugging
  fillSeedHistoDebugger(result.begin(),result.end());

}


TempTrajectory BaseCkfTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{
  TempTrajectory result( seed, seed.direction());
  if (  seed.direction() == alongMomentum) {
    theForwardPropagator = &(*thePropagatorAlong);
    theBackwardPropagator = &(*thePropagatorOpposite);
  }
  else {
    theForwardPropagator = &(*thePropagatorOpposite);
    theBackwardPropagator = &(*thePropagatorAlong);
  }

  std::vector<TM> seedMeas;
  seedMeasurements(seed, seedMeas);
  for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++)
    result.push(*i);            
  return result;
}


bool BaseCkfTrajectoryBuilder::toBeContinued (TempTrajectory& traj) const
{
  //
  // check on sign flip (needs to be done first since trajectory
  // will be invalidated)
  //
  const TempTrajectory::DataContainer & tms = traj.measurements();
  // Check flip in q-significance. The loop over all TMs could be 
  // avoided by storing the current significant q in the trajectory
  if ( theChargeSignificance>0. ) {
    int qSig(0);
    // skip first two hits (don't rely on significance of q/p)
    for( TempTrajectory::DataContainer::size_type itm=2; itm<tms.size(); ++itm ) {
      TrajectoryStateOnSurface tsos = tms[itm].updatedState();
      if ( !tsos.isValid() )  continue;
      double significance = tsos.localParameters().vector()(0) /
	sqrt(tsos.localError().matrix()(0,0));
      // don't deal with measurements compatible with 0
      if ( fabs(significance)<theChargeSignificance )  continue;
      //
      // if charge not yet defined: store first significant Q
      //
      if ( qSig==0 ) {
	qSig = significance>0 ? 1 : -1;
      }
      //
      // else: invalidate and terminate in case of a change of sign
      //
      else {
	if ( (significance<0.&&qSig>0) || (significance>0.&&qSig<0) ) {
	  traj.invalidate();
	  return false;
	}
      }
    }
  }
  // 
  // check on number of lost hits
  //
  if ( traj.lostHits() > theMaxLostHit) return false;

  // check for conscutive lost hits only at the end 
  // (before the last valid hit),
  // since if there was an unacceptable gap before the last 
  // valid hit the trajectory would have been stopped already

  int consecLostHit = 0;

//   const TempTrajectory::DataContainer & tms = traj.measurements();
  //for( TempTrajectory::DataContainer::const_iterator itm=tms.end()-1; itm>=tms.begin(); itm--) {
  for( TempTrajectory::DataContainer::const_iterator itm=tms.rbegin(), itb = tms.rend(); itm != itb; --itm) {
    if (itm->recHit()->isValid()) break;
    else if ( // FIXME: restore this:   !Trajectory::inactive(itm->recHit()->det()) &&
	     Trajectory::lost(*itm->recHit())) consecLostHit++;
  }
  if (consecLostHit > theMaxConsecLostHit) return false; 

  // stopping condition from region has highest priority
  // if ( regionalCondition && !(*regionalCondition)(traj) )  return false;
  // next: pt-cut
  if ( !(*theMinPtCondition)(traj) )  return false;
  if ( !(*theMaxHitsCondition)(traj) )  return false;
  // finally: configurable condition
  // FIXME: restore this:  if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}


 bool BaseCkfTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{
  // check validity (might have been set to false 
  // in charge significance check)
  if ( !traj.isValid() )  return false;

//    cout << "qualityFilter called for trajectory with " 
//         << traj.foundHits() << " found hits and Chi2 = "
//         << traj.chiSquared() << endl;

  if ( traj.foundHits() >= theMinimumNumberOfHits) {
    return true;
  }
  else {
    return false;
  }
}


void 
BaseCkfTrajectoryBuilder::addToResult (TempTrajectory& tmptraj, 
				       TrajectoryContainer& result) const
{
  // quality check
  if ( !qualityFilter(tmptraj) )  return;
  Trajectory traj = tmptraj.toTrajectory();	
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}


BaseCkfTrajectoryBuilder::StateAndLayers
BaseCkfTrajectoryBuilder::findStateAndLayers(const TempTrajectory& traj) const
{
  if (traj.empty())
    {
      //set the currentState to be the one from the trajectory seed starting point
      PTrajectoryStateOnDet ptod = traj.seed().startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theMeasurementTracker->geomTracker()->idToDet(id);                    
      const Surface * surface=&g->surface();
      TrajectoryStateTransform tsTransform;
      
      TSOS currentState = TrajectoryStateOnSurface(tsTransform.transientState(ptod,surface,theForwardPropagator->magneticField()));      
      const DetLayer* lastLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(id);      
      return StateAndLayers(currentState,lastLayer->nextLayers( *currentState.freeState(), traj.direction()) );
    }
  else
    {  
      TSOS currentState = traj.lastMeasurement().updatedState();
      return StateAndLayers(currentState,traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction()) );
    }
}



