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
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"


#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

using namespace std;

CkfTrajectoryBuilder::CkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector& iC):
  CkfTrajectoryBuilder(conf,
                       BaseCkfTrajectoryBuilder::createTrajectoryFilter(conf.getParameter<edm::ParameterSet>("trajectoryFilter"), iC))
{}

CkfTrajectoryBuilder::CkfTrajectoryBuilder(const edm::ParameterSet& conf, TrajectoryFilter *filter):
  BaseCkfTrajectoryBuilder(conf, filter)
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

void CkfTrajectoryBuilder::setEvent_(const edm::Event& event, const edm::EventSetup& iSetup) {  
}

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
  if (theMeasurementTracker == 0) {
      throw cms::Exception("LogicError") << "Asking to create trajectories to an un-initialized CkfTrajectoryBuilder.\nYou have to call clone(const MeasurementTrackerEvent *data) and then call trajectories on it instead.\n";
  }
 
  TempTrajectory && startingTraj = createStartingTrajectory( seed );
  
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
  newCand.reserve(2*theMaxCand);

  
  auto trajCandLess = [&](TempTrajectory const & a, TempTrajectory const & b) {
    return  (a.chiSquared() + a.lostHits()*theLostHitPenalty)  <
    (b.chiSquared() + b.lostHits()*theLostHitPenalty);
  };
  
 
  while ( !candidates.empty()) {

    newCand.clear();
    for (auto traj=candidates.begin(); traj!=candidates.end(); traj++) {
      std::vector<TM> meas;
      findCompatibleMeasurements(*sharedSeed, *traj, meas);

      // --- method for debugging
      if(!analyzeMeasurementsDebugger(*traj,meas,
				      theMeasurementTracker,
				      forwardPropagator(*sharedSeed),theEstimator,
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

	for(auto itm = meas.begin(); itm != last; itm++) {
	  TempTrajectory newTraj = *traj;
	  updateTrajectory( newTraj, std::move(*itm));

	  if ( toBeContinued(newTraj)) {
	    newCand.push_back(std::move(newTraj));  std::push_heap(newCand.begin(),newCand.end(),trajCandLess);
	  }
	  else {
	    if ( qualityFilter(newTraj))  addToResult(sharedSeed, newTraj, result);
	    //// don't know yet
	  }
	}
      }


      /*
      auto trajVal = [&](TempTrajectory const & a) {
      	return  a.chiSquared() + a.lostHits()*theLostHitPenalty;
      };

      // safe (stable?) logig: always sort, kill exceeding only if worse than last to keep
      // if ((int)newCand.size() > theMaxCand) std::cout << "TrajVal " << theMaxCand  << ' ' << newCand.size() << ' ' <<  trajVal(newCand.front());
      int toCut = int(newCand.size()) - int(theMaxCand);
      if (toCut>0) {
        // move largest "toCut" to the end
        for (int i=0; i<toCut; ++i)
          std::pop_heap(newCand.begin(),newCand.end()-i,trajCandLess);
        auto fval = trajVal(newCand.front());
        // remove till equal to highest to keep
        for (int i=0; i<toCut; ++i) {
           if (fval==trajVal(newCand.back())) break;
           newCand.pop_back();
        }
	//assert((int)newCand.size() >= theMaxCand);
	//std::cout << "; " << newCand.size() << ' ' << trajVal(newCand.front())  << " " << trajVal(newCand.back());

	// std::make_heap(newCand.begin(),newCand.end(),trajCandLess);
        // push_heap again the one left
        for (auto iter = newCand.begin()+theMaxCand+1; iter<=newCand.end(); ++iter  )
	  std::push_heap(newCand.begin(),iter,trajCandLess);

	// std::cout << "; " << newCand.size() << ' ' << trajVal(newCand.front())  << " " << trajVal(newCand.back()) << std::endl;
      }

      */

      
      // intermedeate login: always sort,  kill all exceeding
      while ((int)newCand.size() > theMaxCand) {
	std::pop_heap(newCand.begin(),newCand.end(),trajCandLess);
	// if ((int)newCand.size() == theMaxCand+1) std::cout << " " << trajVal(newCand.front())  << " " << trajVal(newCand.back()) << std::endl;
	newCand.pop_back();
       }
      
      /*
      //   original logic: sort only if > theMaxCand, kill all exceeding
      if ((int)newCand.size() > theMaxCand) {
	std::sort( newCand.begin(), newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
	// std::partial_sort( newCand.begin(), newCand.begin()+theMaxCand, newCand.end(), TrajCandLess<TempTrajectory>(theLostHitPenalty));
	std::cout << "TrajVal " << theMaxCand  << ' ' << newCand.size() << ' '
	<< trajVal(newCand.back()) << ' ' << trajVal(newCand[theMaxCand-1]) << ' ' << trajVal(newCand[theMaxCand])  << std::endl;
	newCand.resize(theMaxCand);
      }
      */

    } // end loop on candidates

    std::sort_heap(newCand.begin(),newCand.end(),trajCandLess);
    if (theIntermediateCleaning) IntermediateTrajectoryCleaner::clean(newCand);

    candidates.swap(newCand);
    
    LogDebug("CkfPattern") <<result.size()<<" candidates after "<<nIter++<<" CKF iteration: \n"
			   <<PrintoutHelper::dumpCandidates(result)
			   <<"\n "<<candidates.size()<<" running candidates are: \n"
			   <<PrintoutHelper::dumpCandidates(candidates);

  }
}



void CkfTrajectoryBuilder::updateTrajectory( TempTrajectory& traj,
					     TM && tm) const
{
  auto && predictedState = tm.predictedState();
  auto  && hit = tm.recHit();
  if ( hit->isValid()) {
    auto && upState = theUpdator->update( predictedState, *hit);
    traj.emplace( std::move(predictedState), std::move(upState),
		 std::move(hit), tm.estimate(), tm.layer()); 
  }
  else {
    traj.emplace( std::move(predictedState), std::move(hit), 0, tm.layer());
  }
}


void 
CkfTrajectoryBuilder::findCompatibleMeasurements(const TrajectorySeed&seed,
						 const TempTrajectory& traj, 
						 std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;
  std::pair<TSOS,std::vector<const DetLayer*> > && stateAndLayers = findStateAndLayers(traj);
  if (stateAndLayers.second.empty()) return;

  auto layerBegin = stateAndLayers.second.begin();
  auto layerEnd  = stateAndLayers.second.end();
  LogDebug("CkfPattern")<<"looping on "<< stateAndLayers.second.size()<<" layers.";
  const Propagator *fwdPropagator = forwardPropagator(seed);
  for (auto il = layerBegin;  il != layerEnd; il++) {

    LogDebug("CkfPattern")<<"looping on a layer in findCompatibleMeasurements.\n last layer: "<<traj.lastLayer()<<" current layer: "<<(*il);

    TSOS stateToUse = stateAndLayers.first;
    if unlikely ((*il)==traj.lastLayer()) {
	LogDebug("CkfPattern")<<" self propagating in findCompatibleMeasurements.\n from: \n"<<stateToUse;
	//self navigation case
	// go to a middle point first
	TransverseImpactPointExtrapolator middle;
	GlobalPoint center(0,0,0);
	stateToUse = middle.extrapolate(stateToUse, center, *fwdPropagator);
	
	if (!stateToUse.isValid()) continue;
	LogDebug("CkfPattern")<<"to: "<<stateToUse;
      }
    
    LayerMeasurements layerMeasurements(theMeasurementTracker->measurementTracker(), *theMeasurementTracker);
    std::vector<TrajectoryMeasurement> && tmp = layerMeasurements.measurements((**il),stateToUse, *fwdPropagator, *theEstimator);
    
    if ( !tmp.empty()) {
      if ( result.empty()) result.swap(tmp);
      else {
	// keep one dummy TM at the end, skip the others
	result.insert( result.end()-invalidHits, 
		       std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
      }
      invalidHits++;
    }
  }

  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    std::sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
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

