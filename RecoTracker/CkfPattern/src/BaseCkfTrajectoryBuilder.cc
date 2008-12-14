#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

  
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


BaseCkfTrajectoryBuilder::
BaseCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
			 const TrajectoryStateUpdator*         updator,
			 const Propagator*                     propagatorAlong,
			 const Propagator*                     propagatorOpposite,
			 const Chi2MeasurementEstimatorBase*   estimator,
			 const TransientTrackingRecHitBuilder* recHitBuilder,
			 const MeasurementTracker*             measurementTracker,
			 const TrajectoryFilter*               filter):
  theUpdator(updator),
  thePropagatorAlong(propagatorAlong),thePropagatorOpposite(propagatorOpposite),
  theEstimator(estimator),theTTRHBuilder(recHitBuilder),
  theMeasurementTracker(measurementTracker),
  theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
  theForwardPropagator(0),theBackwardPropagator(0),
  theFilter(filter)
{}
 
BaseCkfTrajectoryBuilder::~BaseCkfTrajectoryBuilder(){
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
  return theFilter->toBeContinued(traj);
}


 bool BaseCkfTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{
  return theFilter->qualityFilter(traj);
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
void 
BaseCkfTrajectoryBuilder::addToResult (TempTrajectory& tmptraj, 
				       TempTrajectoryContainer& result) const
{
  // quality check
  if ( !qualityFilter(tmptraj) )  return;
  // discard latest dummy measurements
  TempTrajectory traj = tmptraj;
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj );
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



