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
#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"

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

void CombinatorialTrajectoryBuilder::
limitedCandidates( Trajectory& startingTraj, 
		   TrajectoryContainer& result)
{
  TrajectoryContainer candidates;
  TrajectoryContainer newCand;
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      std::vector<TM> meas = findCompatibleMeasurements(*traj);
      if ( meas.empty()) {
	if ( qualityFilter( *traj)) addToResult( *traj, result);
      }
      else {
	std::vector<TM>::const_iterator last;
	if ( theAlwaysUseInvalid) last = meas.end();
	else {
	  if (meas.front().recHit()->isValid()) {
	    last = find_if( meas.begin(), meas.end(), RecHitIsInvalid());
	  }
	  else last = meas.end();
	}

	for( std::vector<TM>::const_iterator itm = meas.begin(); 
	     itm != last; itm++) {
	  Trajectory newTraj = *traj;
	  updateTrajectory( newTraj, *itm);

	  if ( toBeContinued(newTraj)) newCand.push_back(newTraj);
	  else {
	    if ( qualityFilter(newTraj)) addToResult( newTraj, result);
	    //// don't know yet
	  }
	}
      }
    
      if ((int)newCand.size() > theMaxCand) {
	sort( newCand.begin(), newCand.end(), TrajCandLess(theLostHitPenalty));
	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
    }

    // FIXME: restore intermediary cleaning
//     if (theIntermediateCleaning) {
//       candidates.clear();
//       candidates = intermediaryClean(newCand);
//     } else {
    candidates.swap(newCand);
//     }
  }
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

 bool CombinatorialTrajectoryBuilder::qualityFilter( const Trajectory& traj)
{

//    cout << "qualityFilter called for trajectory with " 
//         << traj.foundHits() << " found hits and Chi2 = "
//         << traj.chiSquared() << endl;

  if ( traj.foundHits() >= theMinHits) {
    return true;
  }
  else {
    return false;
  }
}


void CombinatorialTrajectoryBuilder::addToResult( Trajectory& traj, 
						  TrajectoryContainer& result)
{
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}

void CombinatorialTrajectoryBuilder::updateTrajectory( Trajectory& traj,
						       const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  const TransientTrackingRecHit* hit = tm.recHit();
 
  if ( hit->isValid()) {
    traj.push( TM( predictedState, theUpdator->update( predictedState, *hit),
		   hit, tm.estimate()));
  }
  else {
    traj.push( TM( predictedState, hit));
  }
}

bool CombinatorialTrajectoryBuilder::toBeContinued (const Trajectory& traj)
{
  if ( traj.lostHits() > theMaxLostHit) return false;

  // check for conscutive lost hits only at the end 
  // (before the last valid hit),
  // since if there was an unacceptable gap before the last 
  // valid hit the trajectory would have been stopped already

  int consecLostHit = 0;
  vector<TM> tms = traj.measurements();
  for( vector<TM>::const_iterator itm=tms.end()-1; itm>=tms.begin(); itm--) {
    if (itm->recHit()->isValid()) break;
    else if ( // FIXME: restore this:   !Trajectory::inactive(itm->recHit()->det()) &&
	      Trajectory::lost(*itm->recHit())) consecLostHit++;
  }
  if (consecLostHit > theMaxConsecLostHit) return false; 

  // stopping condition from region has highest priority
  // if ( regionalCondition && !(*regionalCondition)(traj) )  return false;
  // next: pt-cut
  // FIXME: restore this:  if ( !(*theMinPtCondition)(traj) )  return false;
  // finally: configurable condition
  // FIXME: restore this:  if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}

