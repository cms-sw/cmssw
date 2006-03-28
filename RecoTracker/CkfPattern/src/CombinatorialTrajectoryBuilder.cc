#include "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"

CombinatorialTrajectoryBuilder::
CombinatorialTrajectoryBuilder( const MeasurementTracker* tracker,
				const Propagator* prop,
				const TrajectoryStateUpdator* upd,
				const MeasurementEstimator* est,
				const NavigationSchool* ns) :
    
    theTracker(tracker),
    thePropagator(prop->clone()),
    theUpdator(upd),
    theEstimator(est),
    theNavigationSchool(ns)
{}

CombinatorialTrajectoryBuilder::TrajectoryContainer 
CombinatorialTrajectoryBuilder::trajectories(const TrajectorySeed& seed)
{

  // set the correct navigation
  NavigationSetter setter( *theNavigationSchool);
  
  // set the propagation direction
  thePropagator->setPropagationDirection(seed.direction());

  TrajectoryContainer result;

  // analyseSeed( seed);

  Trajectory startingTraj = createStartingTrajectory( seed);

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition
  limitedCandidates( startingTraj, result);

  // analyseResult(result);

  return result;
}

Trajectory CombinatorialTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction());

  std::vector<TM> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
    }
  }
  return result;
}


#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

std::vector<TrajectoryMeasurement> 
CombinatorialTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TkTransientTrackingRecHitBuilder recHitBuilder( theTracker->geomTracker());
  TrajectoryStateTransform tsTransform;

  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    TransientTrackingRecHit* recHit = recHitBuilder.build(&(*ihit));
    const GeomDet* hitGeomDet = 
      theTracker->geomTracker()->idToDet( ihit->geographicalId());
    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    if (ihit == hitRange.second - 1) {
      // the seed trajectory state should correspond to this hit
      PTrajectoryStateOnDet pState( seed.startingState());
      const GeomDet* gdet = theTracker->geomTracker()->idToDet( DetId(pState.detId()));
      if (&gdet->surface() != &hitGeomDet->surface()) {
	std::cout << "CombinatorialTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit" 
		  << std::endl;
	return std::vector<TrajectoryMeasurement>(); // FIXME: should throw exception
      }

      TSOS updatedState = tsTransform.transientState( pState, &(gdet->surface()), 
						      thePropagator->magneticField());
      result.push_back(TM( invalidState, updatedState, recHit));
    }
    else {
      result.push_back(TM( invalidState, recHit));
    }
  }
  return result;
}

 
