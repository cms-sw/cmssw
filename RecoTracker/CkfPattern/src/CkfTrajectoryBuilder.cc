#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"


#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using namespace std;

CkfTrajectoryBuilder::
  CkfTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* recHitBuilder,
		       const MeasurementTracker*             measurementTracker):

    TrackerTrajectoryBuilder(conf.getParameter<edm::ParameterSet>("TrackerTrajectoryBuilderParameters"),
			     updator, propagatorAlong,propagatorOpposite,
			     estimator, recHitBuilder, measurementTracker)
{
  theMaxCand              = conf.getParameter<int>("maxCand");
  theLostHitPenalty       = conf.getParameter<double>("lostHitPenalty");
  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");
}

void CkfTrajectoryBuilder::setEvent(const edm::Event& event) const
{
  theMeasurementTracker->update(event);
}

CkfTrajectoryBuilder::TrajectoryContainer 
CkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed) const
{  
  TrajectoryContainer result;

  // analyseSeed( seed);

  TempTrajectory startingTraj = createStartingTrajectory( seed );

  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition

  limitedCandidates( startingTraj, result);

  // analyseResult(result);

  return result;
}


void CkfTrajectoryBuilder::
limitedCandidates( TempTrajectory& startingTraj, 
		   TrajectoryContainer& result) const
{
  TempTrajectoryContainer candidates; // = TrajectoryContainer();
  TempTrajectoryContainer newCand; // = TrajectoryContainer();
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      std::vector<TM> meas;
      findCompatibleMeasurements(*traj, meas);

      // --- method for debugging
      if(!analyzeMeasurementsDebugger(*traj,meas,
				      theMeasurementTracker,
				      theForwardPropagator,theEstimator,
				      theTTRHBuilder)) return;
      // ---

      if ( meas.empty()) {
	if ( qualityFilter( *traj)) addToResult( *traj, result);
      }
      else {
	std::vector<TM>::const_iterator last;
	if ( theAlwaysUseInvalidHits) last = meas.end();
	else {
	  if (meas.front().recHit()->isValid()) {
	    last = find_if( meas.begin(), meas.end(), RecHitIsInvalid());
	  }
	  else last = meas.end();
	}

	for( std::vector<TM>::const_iterator itm = meas.begin(); 
	     itm != last; itm++) {
	  TempTrajectory newTraj = *traj;
	  updateTrajectory( newTraj, *itm);

	  if ( toBeContinued(newTraj)) {
	    newCand.push_back(newTraj);
	  }
	  else {
	    if ( qualityFilter(newTraj)) addToResult( newTraj, result);
	    //// don't know yet
	  }
	}
      }
    
      if ((int)newCand.size() > theMaxCand) {
	sort( newCand.begin(), newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
    }

    if (theIntermediateCleaning) IntermediateTrajectoryCleaner::clean(newCand);

    candidates.swap(newCand);
  }
}



void CkfTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
					     const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  TM::ConstRecHitPointer hit = tm.recHit();
 
  if ( hit->isValid()) {
    TM tmp = TM( predictedState, theUpdator->update( predictedState, *hit),
		 hit, tm.estimate(), tm.layer()); 
    traj.push(tmp );
  }
  else {
    traj.push( TM( predictedState, hit, 0, tm.layer()));
  }
}


void 
CkfTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj, 
						  std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;
  std::pair<TSOS,std::vector<const DetLayer*> > stateAndLayers = findStateAndLayers(traj);
  if (stateAndLayers.second.empty()) return;

  vector<const DetLayer*>::iterator layerBegin = stateAndLayers.second.begin();
  vector<const DetLayer*>::iterator layerEnd  = stateAndLayers.second.end();
  for (vector<const DetLayer*>::iterator il = layerBegin; 
       il != layerEnd; il++) {
    vector<TrajectoryMeasurement> tmp = theLayerMeasurements->measurements((**il),stateAndLayers.first, *theForwardPropagator, *theEstimator);

    if ( !tmp.empty()) {
      if ( result.empty()) result = tmp;
      else {
	// keep one dummy TM at the end, skip the others
	result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
  }

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
  }

#ifdef DEBUG_INVALID
  bool afterInvalid = false;
  for (vector<TM>::const_iterator i=result.begin();
       i!=result.end(); i++) {
    if ( ! i->recHit().isValid()) afterInvalid = true;
    if (afterInvalid && i->recHit().isValid()) {
      edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: valid hit after invalid!" ;
    }
  }
#endif

  //analyseMeasurements( result, traj);

}

