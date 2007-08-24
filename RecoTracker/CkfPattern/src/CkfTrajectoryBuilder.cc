#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"


#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"

using namespace std;

CkfTrajectoryBuilder::
  CkfTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* RecHitBuilder,
		       const MeasurementTracker*             measurementTracker):

    theUpdator(updator),thePropagatorAlong(propagatorAlong),
    thePropagatorOpposite(propagatorOpposite),theEstimator(estimator),
    theTTRHBuilder(RecHitBuilder),theMeasurementTracker(measurementTracker),
    theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
    theForwardPropagator(0), theBackwardPropagator(0),
    theMinPtCondition(new MinPtTrajectoryFilter(conf.getParameter<double>("ptCut"))),
    theMaxHitsCondition(new MaxHitsTrajectoryFilter(conf.getParameter<int>("maxNumberOfHits")))
{
  theMaxCand              = conf.getParameter<int>("maxCand");
  theMaxLostHit           = conf.getParameter<int>("maxLostHit");
  theMaxConsecLostHit     = conf.getParameter<int>("maxConsecLostHit");
  theLostHitPenalty       = conf.getParameter<double>("lostHitPenalty");
  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
  theMinimumNumberOfHits  = conf.getParameter<int>("minimumNumberOfHits");
  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");
}

CkfTrajectoryBuilder::~CkfTrajectoryBuilder()
{
  delete theLayerMeasurements;
  delete theMinPtCondition;
  delete theMaxHitsCondition;
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

TempTrajectory CkfTrajectoryBuilder::
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



#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

void
CkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed,  std::vector<TrajectoryMeasurement> & result) const
{
 
  TrajectoryStateTransform tsTransform;

  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    //RC TransientTrackingRecHit* recHit = TTRHbuilder->build(&(*ihit));
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

 bool CkfTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
{

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


void CkfTrajectoryBuilder::addToResult( TempTrajectory& tmptraj, 
					TrajectoryContainer& result) const
{
  Trajectory traj = tmptraj.toTrajectory();
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
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

bool CkfTrajectoryBuilder::toBeContinued (const TempTrajectory& traj) const
{
  if ( traj.lostHits() > theMaxLostHit) return false;

  // check for conscutive lost hits only at the end 
  // (before the last valid hit),
  // since if there was an unacceptable gap before the last 
  // valid hit the trajectory would have been stopped already

  int consecLostHit = 0;

  const TempTrajectory::DataContainer & tms = traj.measurements();
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

void 
CkfTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj, 
						  std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;

  TSOS currentState( traj.lastMeasurement().updatedState());

  vector<const DetLayer*> nl = 
    traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
  
  if (nl.empty()) return;

  for (vector<const DetLayer*>::iterator il = nl.begin(); 
       il != nl.end(); il++) {
    vector<TM> tmp = 
      theLayerMeasurements->measurements((**il),currentState, *theForwardPropagator, *theEstimator);

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

