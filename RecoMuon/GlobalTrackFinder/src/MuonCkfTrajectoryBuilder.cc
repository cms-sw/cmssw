//JR#include "RecoTracker/CkfPattern/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoMuon/GlobalTrackFinder/interface/MuonCkfTrajectoryBuilder.h"

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

MuonCkfTrajectoryBuilder::
  MuonCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
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

MuonCkfTrajectoryBuilder::~MuonCkfTrajectoryBuilder()
{
  delete theLayerMeasurements;
  delete theMinPtCondition;
  delete theMaxHitsCondition;
}

void MuonCkfTrajectoryBuilder::setEvent(const edm::Event& event) const
{
  theMeasurementTracker->update(event);
}

MuonCkfTrajectoryBuilder::TrajectoryContainer 
MuonCkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed) const
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

TempTrajectory MuonCkfTrajectoryBuilder::
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

  std::vector<TM> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);            
    }
  }
  return result;
}

void MuonCkfTrajectoryBuilder::
limitedCandidates( TempTrajectory& startingTraj, 
		   TrajectoryContainer& result) const
{
  TempTrajectoryContainer candidates; // = TrajectoryContainer();
  TempTrajectoryContainer newCand; // = TrajectoryContainer();
  candidates.push_back( startingTraj);

  while (!candidates.empty()) {

    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      std::vector<TM> meas = findCompatibleMeasurements(*traj);

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

std::vector<TrajectoryMeasurement> 
MuonCkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
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
	edm::LogError("CkfPattern") << "MuonCkfTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
	return std::vector<TrajectoryMeasurement>(); // FIXME: should throw exception
      }

      TSOS updatedState = tsTransform.transientState( pState, &(gdet->surface()), 
						      theForwardPropagator->magneticField());
      result.push_back(TM( invalidState, updatedState, recHit, 0, hitLayer));
    }
    else {
      //----------- just a test to make the Smoother to work -----------
      PTrajectoryStateOnDet pState( seed.startingState());
      TSOS outerState = tsTransform.transientState( pState, &(hitGeomDet->surface()), 
						    theForwardPropagator->magneticField());
      TSOS innerState   = theBackwardPropagator->propagate(outerState,hitGeomDet->surface());
      TSOS innerUpdated = theUpdator->update(innerState,*recHit);

      result.push_back(TM( invalidState, innerUpdated, recHit, 0, hitLayer));
      //-------------------------------------------------------------

      //result.push_back(TM( invalidState, recHit, 0, hitLayer));
    }
  }

  // method for debugging
  fillSeedHistoDebugger(result.begin(),result.end());

  return result;
}

 bool MuonCkfTrajectoryBuilder::qualityFilter( const TempTrajectory& traj) const
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


void MuonCkfTrajectoryBuilder::addToResult( TempTrajectory& tmptraj, 
					TrajectoryContainer& result) const
{
  Trajectory traj = tmptraj.toTrajectory();
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}

void MuonCkfTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
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

bool MuonCkfTrajectoryBuilder::toBeContinued (const TempTrajectory& traj) const
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


void MuonCkfTrajectoryBuilder::collectMeasurement(const std::vector<const DetLayer*>& nl,const TrajectoryStateOnSurface & currentState, std::vector<TM>& result,int& invalidHits) const{
  for (vector<const DetLayer*>::const_iterator il = nl.begin();
       il != nl.end(); il++) {
    vector<TM> tmp =
      theLayerMeasurements->measurements((**il),currentState, *theForwardPropagator, *theEstimator);

    LogDebug("CkfPattern")<<tmp.size()<<" measurements returned by LayerMeasurements";

    if ( !tmp.empty()) {
      // FIXME durty-durty-durty cleaning: never do that please !
      /*      for (vector<TM>::iterator it = tmp.begin(); it!=tmp.end(); ++it)
              {if (it->recHit()->det()==0) it=tmp.erase(it)--;}*/

      if ( result.empty()) result = tmp;
      else {
        // keep one dummy TM at the end, skip the others
        result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
  }

  LogDebug("CkfPattern")<<result.size()<<" total measurements";
  for (vector<TrajectoryMeasurement>::iterator it = result.begin(); it!=result.end();++it){
    LogDebug("CkfPattern")<<"layer pointer: "<<it->layer()<<"\n"
                          <<"estimate: "<<it->estimate()<<"\n"
                          <<"forward state: \n"<<it->forwardPredictedState()
                          <<"geomdet pointer from rechit: "<<it->recHit()->det()<<"\n"
                          <<"detId: "<<it->recHit()->geographicalId().rawId();
  }

}



std::vector<TrajectoryMeasurement> 
MuonCkfTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj) const
{
  vector<TM> result;
  int invalidHits = 0;

  //----JR--- 15 March 2007  ---START MODIF
  vector<const DetLayer*> nl;
  TSOS currentState;

  if (traj.empty())
    {
      edm::LogInfo("CkfPattern")<<"using JR patch for no measurement case";
      //what if there are no measurement on the Trajectory

      //set the currentState to be the one from the trajectory seed starting point
      PTrajectoryStateOnDet ptod =traj.seed().startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theMeasurementTracker->geomTracker()->idToDet(id);
      const Surface * surface=&g->surface();
      TrajectoryStateTransform tsTransform;
      currentState = tsTransform.transientState(ptod,surface,theForwardPropagator->magneticField());

      //set the next layers to be that one the state is on
      const DetLayer * l=theMeasurementTracker->geometricSearchTracker()->detLayer(id);
      if ( traj.direction() == alongMomentum ){
        //will fail if the building is outside-in
        //because the propagator will cross over the barrel and give measurement on the other side of the barrel
        nl.clear();
        nl.push_back(l);
        collectMeasurement(nl,currentState,result,invalidHits);
      }

      if (result.size()==0)
        {
	  edm::LogInfo("CkfPattern")<<"using JR patch: need to go to next layer to get measurements";
          //the following will "JUMP" the first layer measurements
          nl = l->nextLayers(*currentState.freeState(), traj.direction());
          invalidHits=0;
          collectMeasurement(nl,currentState,result,invalidHits);}
    }
  else
    {
      currentState = traj.lastMeasurement().updatedState();
      nl = traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
      if (nl.empty()){LogDebug("CkfPattern")<<" no next layers... going "<<traj.direction()<<"\n from: \n"<<currentState<<"\n from detId: "<<traj.lastMeasurement().recHit()->geographicalId().rawId(); return result;}

      collectMeasurement(nl,currentState,result,invalidHits);
    }
  //----JR--- 15 March 2007  ---END


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
      edm::LogError("CkfPattern") << "MuonCkfTrajectoryBuilder error: valid hit after invalid!" ;
    }
  }
#endif

  //analyseMeasurements( result, traj);

  return result;
}

