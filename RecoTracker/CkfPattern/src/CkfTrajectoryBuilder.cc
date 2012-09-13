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
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

#include "FWCore/Services/interface/UpdaterService.h"

using namespace std;


CkfTrajectoryBuilder::
  CkfTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* recHitBuilder,
		       const MeasurementTracker*             measurementTracker,
		       const TrajectoryFilter*               filter):

    BaseCkfTrajectoryBuilder(conf,
			     updator, propagatorAlong,propagatorOpposite,
			     estimator, recHitBuilder, measurementTracker,filter)
{
  theMaxCand              = conf.getParameter<int>("maxCand");
  theLostHitPenalty       = conf.getParameter<double>("lostHitPenalty");
  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");
  /*
    theSharedSeedCheck = conf.getParameter<bool>("SharedSeedCheck");
    std::stringstream ss;
    ss<<"CkfTrajectoryBuilder_"<<conf.getParameter<std::string>("ComponentName")<<"_"<<this;
    theUniqueName = ss.str();
    LogDebug("CkfPattern")<<"my unique name is: "<<theUniqueName;
  */
}

/*
  void CkfTrajectoryBuilder::setEvent(const edm::Event& event) const
  {
  theMeasurementTracker->update(event);
  }
*/

CkfTrajectoryBuilder::TrajectoryContainer 
CkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed) const
{  
  TrajectoryContainer result;
  result.reserve(5);
  trajectories(seed, result);
  return result;
}

/*
  void CkfTrajectoryBuilder::rememberSeedAndTrajectories(const TrajectorySeed& seed,
  CkfTrajectoryBuilder::TrajectoryContainer &result) const
  {
  
  //result ----> theCachedTrajectories
  //every first iteration on event. forget about everything that happened before
  if (edm::Service<UpdaterService>()->checkOnce(theUniqueName)) 
  theCachedTrajectories.clear();
  
  if (seed.nHits()==0) return;
  
  //then remember those trajectories
  for (TrajectoryContainer::iterator traj=result.begin();
  traj!=result.end(); ++traj) {
  theCachedTrajectories.insert(std::make_pair(seed.recHits().first->geographicalId(),*traj));
  }  
  }
  
  bool CkfTrajectoryBuilder::sharedSeed(const TrajectorySeed& s1,const TrajectorySeed& s2) const{
  //quit right away on nH=0
  if (s1.nHits()==0 || s2.nHits()==0) return false;
  //quit right away if not the same number of hits
  if (s1.nHits()!=s2.nHits()) return false;
  TrajectorySeed::range r1=s1.recHits();
  TrajectorySeed::range r2=s2.recHits();
  TrajectorySeed::const_iterator i1,i2;
  TrajectorySeed::const_iterator & i1_e=r1.second,&i2_e=r2.second;
  TrajectorySeed::const_iterator & i1_b=r1.first,&i2_b=r2.first;
  //quit right away if first detId does not match. front exist because of ==0 ->quit test
  if(i1_b->geographicalId() != i2_b->geographicalId()) return false;
  //then check hit by hit if they are the same
  for (i1=i1_b,i2=i2_b;i1!=i1_e,i2!=i2_e;++i1,++i2){
  if (!i1->sharesInput(&(*i2),TrackingRecHit::all)) return false;
  }
  return true;
  }
  bool CkfTrajectoryBuilder::seedAlreadyUsed(const TrajectorySeed& seed,
  CkfTrajectoryBuilder::TempTrajectoryContainer &candidates) const
  {
  //theCachedTrajectories ---> candidates
  if (seed.nHits()==0) return false;
  bool answer=false;
  pair<SharedTrajectory::const_iterator, SharedTrajectory::const_iterator> range = 
  theCachedTrajectories.equal_range(seed.recHits().first->geographicalId());
  SharedTrajectory::const_iterator trajP;
  for (trajP = range.first; trajP!=range.second;++trajP){
  //check whether seeds are identical     
  if (sharedSeed(trajP->second.seed(),seed)){
  candidates.push_back(trajP->second);
  answer=true;
  }//already existing trajectory shares the seed.   
  }//loop already made trajectories      
  
  return answer;
  }
*/

void
CkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed, CkfTrajectoryBuilder::TrajectoryContainer &result) const
{  
  // analyseSeed( seed);
  /*
    if (theSharedSeedCheck){
    TempTrajectoryContainer candidates;
    if (seedAlreadyUsed(seed,candidates))
    {
    //start with those candidates already made before
    limitedCandidates(candidates,result);
    //and quit
    return;
    }
    }
  */

  buildTrajectories(seed, result,nullptr);
}

TempTrajectory CkfTrajectoryBuilder::buildTrajectories (const TrajectorySeed&seed,
							TrajectoryContainer &result,
							const TrajectoryFilter*) const {
  
  TempTrajectory startingTraj = createStartingTrajectory( seed );
  
  /// limitedCandidates( startingTraj, regionalCondition, result);
  /// FIXME: restore regionalCondition
  limitedCandidates(seed, startingTraj, result);
  
  return startingTraj;

  /*
  //and remember what you just did
  if (theSharedSeedCheck)  rememberSeedAndTrajectories(seed,result);
  */
  
  // analyseResult(result);
}

void CkfTrajectoryBuilder::
limitedCandidates(const TrajectorySeed&seed, TempTrajectory& startingTraj,
		   TrajectoryContainer& result) const
{
  TempTrajectoryContainer candidates;
  candidates.push_back( startingTraj);
  boost::shared_ptr<const TrajectorySeed>  sharedSeed(new TrajectorySeed(seed));
  limitedCandidates(sharedSeed, candidates,result);
}

void CkfTrajectoryBuilder::
limitedCandidates(const boost::shared_ptr<const TrajectorySeed> & sharedSeed, TempTrajectoryContainer &candidates,
		   TrajectoryContainer& result) const
{
  unsigned int nIter=1;
  TempTrajectoryContainer newCand; // = TrajectoryContainer();

 
  while ( !candidates.empty()) {

    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      std::vector<TM> meas;
      findCompatibleMeasurements(*sharedSeed, *traj, meas);

      // --- method for debugging
      if(!analyzeMeasurementsDebugger(*traj,meas,
				      theMeasurementTracker,
				      theForwardPropagator,theEstimator,
				      theTTRHBuilder)) return;
      // ---

      if ( meas.empty()) {
	if ( qualityFilter( *traj)) addToResult(sharedSeed, *traj, result);
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
	    if ( qualityFilter(newTraj)) addToResult(sharedSeed, newTraj, result);
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
    
    LogDebug("CkfPattern") <<result.size()<<" candidates after "<<nIter++<<" CKF iteration: \n"
			   <<PrintoutHelper::dumpCandidates(result)
			   <<"\n "<<candidates.size()<<" running candidates are: \n"
			   <<PrintoutHelper::dumpCandidates(candidates);

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
CkfTrajectoryBuilder::findCompatibleMeasurements(const TrajectorySeed&seed,
						 const TempTrajectory& traj, 
						 std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;
  std::pair<TSOS,std::vector<const DetLayer*> > stateAndLayers = findStateAndLayers(traj);
  if (stateAndLayers.second.empty()) return;

  vector<const DetLayer*>::iterator layerBegin = stateAndLayers.second.begin();
  vector<const DetLayer*>::iterator layerEnd  = stateAndLayers.second.end();
  LogDebug("CkfPattern")<<"looping on "<< stateAndLayers.second.size()<<" layers.";
  for (vector<const DetLayer*>::iterator il = layerBegin; 
       il != layerEnd; il++) {

    LogDebug("CkfPattern")<<"looping on a layer in findCompatibleMeasurements.\n last layer: "<<traj.lastLayer()<<" current layer: "<<(*il);

    TSOS stateToUse = stateAndLayers.first;
    if ((*il)==traj.lastLayer())
      {
	LogDebug("CkfPattern")<<" self propagating in findCompatibleMeasurements.\n from: \n"<<stateToUse;
	//self navigation case
	// go to a middle point first
	TransverseImpactPointExtrapolator middle;
	GlobalPoint center(0,0,0);
	stateToUse = middle.extrapolate(stateToUse, center, *theForwardPropagator);
	
	if (!stateToUse.isValid()) continue;
	LogDebug("CkfPattern")<<"to: "<<stateToUse;
      }
    
    vector<TrajectoryMeasurement> tmp = theLayerMeasurements->measurements((**il),stateToUse, *theForwardPropagator, *theEstimator);
    
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

  LogDebug("CkfPattern")<<"starting from:\n"
			<<"x: "<<stateAndLayers.first.globalPosition()<<"\n"
			<<"p: "<<stateAndLayers.first.globalMomentum()<<"\n"
			<<PrintoutHelper::dumpMeasurements(result);

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

