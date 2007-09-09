#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TrajectorySegmentBuilder.h"


#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "RecoTracker/CkfPattern/interface/GroupedTrajCandLess.h"
#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/RegionalTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// only included for RecHit comparison operator:
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include <algorithm> 

using namespace std;

//#define DBG_GCTB
//#define DBG2_GCTB

//#define STANDARD_INTERMEDIARYCLEAN

#ifdef STANDARD_INTERMEDIARYCLEAN
#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#endif

/* ====== B.M. to be ported layer ===========
#ifdef DBG_GCTB
#include "RecoTracker/CkfPattern/src/ShowCand.h"
#endif
// #define DBG2_GCTB
#ifdef DBG2_GCTB
#include "RecoTracker/CkfPattern/src/SimIdPrinter.h"
#include "Tracker/TkDebugTools/interface/LayerFinderByDet.h"
#include "Tracker/TkLayout/interface/TkLayerName.h"
#endif
=================================== */


GroupedCkfTrajectoryBuilder::
GroupedCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
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
  // fill data members from parameters (eventually data members could be dropped)
  //
  theMaxCand                  = conf.getParameter<int>("maxCand");

  theLostHitPenalty           = conf.getParameter<double>("lostHitPenalty");
  theFoundHitBonus            = conf.getParameter<double>("foundHitBonus");
  theIntermediateCleaning     = conf.getParameter<bool>("intermediateCleaning");
  theAlwaysUseInvalid         = conf.getParameter<bool>("alwaysUseInvalidHits");
  theLockHits                 = conf.getParameter<bool>("lockHits");
  theBestHitOnly              = conf.getParameter<bool>("bestHitOnly");
  theMinNrOf2dHitsForRebuild  = 2;
  theRequireSeedHitsInRebuild = conf.getParameter<bool>("requireSeedHitsInRebuild");
  theMinNrOfHitsForRebuild    = max(0,conf.getParameter<int>("minNrOfHitsForRebuild"));

  /* ======= B.M. to be ported layer ===========
  bool setOK = thePropagator->setMaxDirectionChange(1.6);
  if (!setOK) 
    cout  << "GroupedCkfTrajectoryBuilder WARNING: "
	  << "propagator does not support setMaxDirectionChange" 
	  << endl;
  //   addStopCondition(theMinPtStopCondition);

  theConfigurableCondition = createAlgo<TrajectoryFilter>(componentConfig("StopCondition"));
  ===================================== */

}


void GroupedCkfTrajectoryBuilder::setEvent(const edm::Event& event) const
{
  theMeasurementTracker->update(event);
}

GroupedCkfTrajectoryBuilder::TrajectoryContainer 
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed) const 
{
  return buildTrajectories(seed,0);
}

GroupedCkfTrajectoryBuilder::TrajectoryContainer 
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed, 
					   const TrackingRegion& region) const
{
  RegionalTrajectoryFilter regionalCondition(region);
  return buildTrajectories(seed,&regionalCondition);
}

GroupedCkfTrajectoryBuilder::TrajectoryContainer 
GroupedCkfTrajectoryBuilder::buildTrajectories (const TrajectorySeed& seed,
						const TrajectoryFilter* regionalCondition) const
{
  //B.M. TimeMe tm("GroupedCkfTrajectoryBuilder", false);

  
  // set the propagation direction
  //B.M. thePropagator->setPropagationDirection(seed.direction());

  TrajectoryContainer result;

  analyseSeed( seed);

  TempTrajectory startingTraj = createStartingTrajectory( seed);

  groupedLimitedCandidates( startingTraj, regionalCondition, theForwardPropagator, result);
  if ( result.empty() )  return result;


  //
  // try to additional hits in the seeding region
  //
  if ( theMinNrOfHitsForRebuild>0 ) {
    // reverse direction
    //thePropagator->setPropagationDirection(oppositeDirection(seed.direction()));
    // rebuild part of the trajectory
    rebuildSeedingRegion(startingTraj,result);
  }
  analyseResult(result);

#ifdef DBG_GCTB
  cout << "GroupedCkfTrajectoryBuilder: returning result of size " << result.size() << endl;
#endif

  return result;
}



void 
GroupedCkfTrajectoryBuilder::groupedLimitedCandidates (TempTrajectory& startingTraj, 
						       const TrajectoryFilter* regionalCondition,
						       const Propagator* propagator, 
						       TrajectoryContainer& result) const
{
  TempTrajectoryContainer candidates;
  TempTrajectoryContainer newCand;
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      if ( !advanceOneLayer(*traj,regionalCondition, propagator, newCand,result) ) {
#ifdef DBG_GCTB
	cout << "GCTB: terminating after advanceOneLayer==false" << endl;
#endif
 	continue;
      }
    
#ifdef DBG_GCTB
      cout << "newCand(1)";
      for ( TempTrajectoryContainer::const_iterator it=newCand.begin();
	    it!=newCand.end(); it++ ) 
	cout << " " << it->lostHits() << " " << it->foundHits() 
	     << " " << it->chiSquared() << " ;";
      cout << endl;
//       cout << "newCand.size() = " << newCand.size() << endl;
#endif
      if ((int)newCand.size() > theMaxCand) {
#ifdef DBG_GCTB
	//ShowCand()(newCand);
#endif
 	sort( newCand.begin(), newCand.end(), GroupedTrajCandLess(theLostHitPenalty,theFoundHitBonus));
 	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
#ifdef DBG_GCTB
      cout << "newCand(2)";
      for ( TempTrajectoryContainer::const_iterator it=newCand.begin();
	    it!=newCand.end(); it++ ) 
	cout << " " << it->lostHits() << " " << it->foundHits() 
	     << " " << it->chiSquared() << " ;";
      cout << endl;
#endif
    }

#ifdef DBG_GCTB
    cout << "newCand.size() at end = " << newCand.size() << endl;
#endif
/*
    if (theIntermediateCleaning) {
      candidates.clear();
      candidates = groupedIntermediaryClean(newCand);
    } else {
      candidates.swap(newCand);
    }
*/
    if (theIntermediateCleaning) {
#ifdef STANDARD_INTERMEDIARYCLEAN
	IntermediateTrajectoryCleaner::clean(newCand);	
#else 
	groupedIntermediaryClean(newCand);
#endif	

    }	
    candidates.swap(newCand);

#ifdef DBG_GCTB
    cout << "candidates(3)";
    for ( TempTrajectoryContainer::const_iterator it=candidates.begin();
	  it!=candidates.end(); it++ ) 
      cout << " " << it->lostHits() << " " << it->foundHits() 
	   << " " << it->chiSquared() << " ;";
    cout << endl;

    cout << "after intermediate cleaning = " << candidates.size() << endl;
    //B.M. ShowCand()(candidates);
#endif
  }
}

bool 
GroupedCkfTrajectoryBuilder::advanceOneLayer (TempTrajectory& traj, 
					      const TrajectoryFilter* regionalCondition, 
					      const Propagator* propagator,
					      TempTrajectoryContainer& newCand, 
					      TrajectoryContainer& result) const
{
  std::pair<TSOS,std::vector<const DetLayer*> > stateAndLayers = findStateAndLayers(traj);
  vector<const DetLayer*>::iterator layerBegin = stateAndLayers.second.begin();
  vector<const DetLayer*>::iterator layerEnd   = stateAndLayers.second.end();

  //   if (nl.empty()) {
  //     addToResult(traj,result);
  //     return false;
  //   }
  
#ifdef DBG_GCTB
  #include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
  #include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
  //B.M. TkLayerName layerName;
  //B.M. cout << "Started from " << layerName(traj.lastLayer()) 
  const BarrelDetLayer* sbdl = dynamic_cast<const BarrelDetLayer*>(traj.lastLayer());
  const ForwardDetLayer* sfdl = dynamic_cast<const ForwardDetLayer*>(traj.lastLayer());
  if (sbdl) cout << "Started from " << traj.lastLayer() << " r " << sbdl->specificSurface().radius() << endl;
  if (sfdl) cout << "Started from " << traj.lastLayer() << " z " << sfdl->specificSurface().position().z() << endl;
  cout << "Trying to go to";
  for ( vector<const DetLayer*>::iterator il=nl.begin();
        il!=nl.end(); il++){ 
    //B.M. cout << " " << layerName(*il)  << " " << *il << endl;
    const BarrelDetLayer* bdl = dynamic_cast<const BarrelDetLayer*>(*il);
    const ForwardDetLayer* fdl = dynamic_cast<const ForwardDetLayer*>(*il);

    if (bdl) cout << " r " << bdl->specificSurface().radius() << endl;
    if (fdl) cout << " z " << fdl->specificSurface().position().z() << endl;
    //cout << " " << *il << endl;   
  }
  cout << endl;
#endif

  bool foundSegments(false);
  bool foundNewCandidates(false);
  for ( vector<const DetLayer*>::iterator il=layerBegin; 
	il!=layerEnd; il++) {
    TrajectorySegmentBuilder layerBuilder(theMeasurementTracker,
					  theLayerMeasurements,
					  **il,*propagator,
					  *theUpdator,*theEstimator,
					  theLockHits,theBestHitOnly);
    
#ifdef DBG_GCTB
    cout << "GCTB: starting from r / z = " << stateAndLayers.first.globalPosition().perp()
	 << " / " << stateAndLayers.first.globalPosition().z() << " , pt / pz = " 
	 << stateAndLayers.first.globalMomentum().perp() << " / " 
	 << stateAndLayers.first.globalMomentum().z() << " for layer at "
	 << *il << endl;
    cout << "     errors:";
    for ( int i=0; i<5; i++ )  cout << " " << sqrt(stateAndLayers.first.curvilinearError().matrix()(i,i));
    cout << endl;
#endif

    TrajectoryContainer segments=
      layerBuilder.segments(stateAndLayers.first);

#ifdef DBG_GCTB
    cout << "GCTB: number of segments = " << segments.size() << endl;
#endif

    if ( !segments.empty() )  foundSegments = true;

    for ( TrajectoryContainer::const_iterator is=segments.begin();
	  is!=segments.end(); is++ ) {
      //
      // assume "invalid hit only" segment is last in list
      //
      vector<TM> measurements(is->measurements());
      if ( !theAlwaysUseInvalid && is!=segments.begin() && measurements.size()==1 && 
	   !measurements.front().recHit()->isValid() )  break;
      //
      // create new candidate
      //
      TempTrajectory newTraj(traj);
      for ( vector<TM>::const_iterator im=measurements.begin();
	    im!=measurements.end(); im++ )  newTraj.push(*im);
      //if ( toBeContinued(newTraj,regionalCondition) ) { TOBE FIXED
      if ( toBeContinued(newTraj) ) {

#ifdef DBG_GCTB
	cout << "GCTB: adding new trajectory to candidates" << endl;
#endif

	newCand.push_back(newTraj);
	foundNewCandidates = true;
      }
      else {

#ifdef DBG_GCTB
	cout << "GCTB: adding new trajectory to results" << endl;
#endif

	addToResult(newTraj,result);
      }
    }
  }

#ifdef DBG_GCTB
  if ( !foundSegments )  cout << "GCTB: adding input trajectory to result" << endl;
#endif

  if ( !foundSegments )  addToResult(traj,result);
  return foundNewCandidates;
}

//TempTrajectoryContainer
void
GroupedCkfTrajectoryBuilder::groupedIntermediaryClean (TempTrajectoryContainer& theTrajectories) const 
{
  //if (theTrajectories.empty()) return TrajectoryContainer();
  //TrajectoryContainer result;
  if (theTrajectories.empty()) return;  
  //RecHitEqualByChannels recHitEqualByChannels(false, false);
  int firstLayerSize, secondLayerSize;

  for (TempTrajectoryContainer::iterator firstTraj=theTrajectories.begin();
       firstTraj!=(theTrajectories.end()-1); firstTraj++) {

    if ( (!firstTraj->isValid()) ||
         (!firstTraj->lastMeasurement().recHit()->isValid()) ) continue;
    const TempTrajectory::DataContainer & firstMeasurements = firstTraj->measurements();
    vector<const DetLayer*> firstLayers = layers(firstMeasurements);
    firstLayerSize = firstLayers.size();
    if ( firstLayerSize<4 )  continue;

    for (TempTrajectoryContainer::iterator secondTraj=(firstTraj+1);
       secondTraj!=theTrajectories.end(); secondTraj++) {

      if ( (!secondTraj->isValid()) ||
           (!secondTraj->lastMeasurement().recHit()->isValid()) ) continue;
      const TempTrajectory::DataContainer & secondMeasurements = secondTraj->measurements();
      vector<const DetLayer*> secondLayers = layers(secondMeasurements);
      secondLayerSize = secondLayers.size();
      //
      // only candidates using the same last 3 layers are compared
      //
      if ( firstLayerSize!=secondLayerSize )  continue;
      if ( firstLayers[0]!=secondLayers[0] ||
	   firstLayers[1]!=secondLayers[1] ||
	   firstLayers[2]!=secondLayers[2] )  continue;

      TempTrajectory::DataContainer::const_iterator im1 = firstMeasurements.rbegin();
      TempTrajectory::DataContainer::const_iterator im2 = secondMeasurements.rbegin();
      //
      // check for identical hits in the last layer
      //
      bool unequal(false);
      const DetLayer* layerPtr = firstLayers[0];
      while ( im1!=firstMeasurements.rend()&&im2!=secondMeasurements.rend() ) {
	if ( im1->layer()!=layerPtr || im2->layer()!=layerPtr )  break;
	if ( !(im1->recHit()->isValid()) || !(im2->recHit()->isValid()) ||
	     //!recHitEqualByChannels(im1->recHit(),im2->recHit()) ) {
	     !im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::some) ) {
	  unequal = true;
	  break;
	}
	--im1;
	--im2;
      }
      if ( im1==firstMeasurements.rend() || im2==secondMeasurements.rend() ||
	   im1->layer()==layerPtr || im2->layer()==layerPtr || unequal )  continue;
      //
      // check for invalid hits in the layer -2
      // compare only candidates with invalid / valid combination
      //
      layerPtr = firstLayers[1];
      bool firstValid(true);
      while ( im1!=firstMeasurements.rend()&&im1->layer()==layerPtr ) {
	if ( !im1->recHit()->isValid() )  firstValid = false;
	--im1;
      }
      bool secondValid(true);
      while ( im2!=secondMeasurements.rend()&&im2->layer()==layerPtr ) {
	if ( !im2->recHit()->isValid() )  secondValid = false;
	--im2;
      }
      if ( !tkxor(firstValid,secondValid) )  continue;
      //
      // ask for identical hits in layer -3
      //
      unequal = false;
      layerPtr = firstLayers[2];
      while ( im1!=firstMeasurements.rend()&&im2!=secondMeasurements.rend() ) {
	if ( im1->layer()!=layerPtr || im2->layer()!=layerPtr )  break;
	if ( !(im1->recHit()->isValid()) || !(im2->recHit()->isValid()) ||
	     //!recHitEqualByChannels(im1->recHit(),im2->recHit()) ) {
	     !im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::some) ) {
	  unequal = true;
	  break;
	}
	--im1;
	--im2;
      }
      if ( im1==firstMeasurements.rend() || im2==secondMeasurements.rend() ||
	   im1->layer()==layerPtr || im2->layer()==layerPtr || unequal )  continue;

      if ( !firstValid ) {
	firstTraj->invalidate();
	break;
      }
      else {
	secondTraj->invalidate();
	break;
      }
    }
  }
/*
  for (TempTrajectoryContainer::const_iterator it = theTrajectories.begin();
       it != theTrajectories.end(); it++) {
    if(it->isValid()) result.push_back( *it);
  }

  return result;
*/
  theTrajectories.erase(std::remove_if( theTrajectories.begin(),theTrajectories.end(),
                                        std::not1(std::mem_fun_ref(&TempTrajectory::isValid))),
 //                                     boost::bind(&TempTrajectory::isValid,_1)), 
                        theTrajectories.end());
}

vector<const DetLayer*>
GroupedCkfTrajectoryBuilder::layers (const TempTrajectory::DataContainer& measurements) const 
{
  //layer measurements are sorted from last to first
  vector<const DetLayer*> result;
  if ( measurements.empty() )  return result;

  result.push_back(measurements.back().layer());
  TempTrajectory::DataContainer::const_iterator ifirst = measurements.rbegin();
  --ifirst;	 
  for ( TempTrajectory::DataContainer::const_iterator im=ifirst;
	im!=measurements.rend(); --im ) {
    if ( im->layer()!=result.back() )  result.push_back(im->layer());
  }
#ifdef DBG2_GCTB
  for (vector<const DetLayer*>::const_iterator iter = result.begin(); iter != result.end(); iter++){
	if (!*iter) cout << "Warning: null det layer!! " << endl;
  }
#endif
  return result;
}

void
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion 
(TempTrajectory& startingTraj, TrajectoryContainer& result) const
{
  //
  // Rebuilding of trajectories. Candidates are taken from result,
  // which will be replaced with the solutions after rebuild
  // (assume vector::swap is more efficient than building new container)
  //
#ifdef DBG2_GCTB
  cout << "Starting to rebuild " << result.size() << " tracks" << endl;
#endif
  //
  // Fitter (need to create it here since the propagation direction
  // might change between different starting trajectories)
  //
  KFTrajectoryFitter fitter(*theBackwardPropagator,updator(),estimator());
  //
  TempTrajectoryContainer reFitted;
  TrajectorySeed::range rseedHits = startingTraj.seed().recHits();
  std::vector<const TrackingRecHit*> seedHits;
  //seedHits.insert(seedHits.end(), rseedHits.first, rseedHits.second);
  //for (TrajectorySeed::recHitContainer::const_iterator iter = rseedHits.first; iter != rseedHits.second; iter++){
  //	seedHits.push_back(&*iter);
  //}

  //unsigned int nSeed(seedHits.size());
  unsigned int nSeed(rseedHits.second-rseedHits.first);
  //seedHits.reserve(nSeed);
  TrajectoryContainer rebuiltTrajectories;
  for ( TrajectoryContainer::iterator it=result.begin();
	it!=result.end(); it++ ) {
    //
    // skip candidates which are not exceeding the seed size
    // (should not happen) - keep existing trajectory
    //
    if ( it->measurements().size()<=startingTraj.measurements().size() ) {
      rebuiltTrajectories.push_back(*it);
      #ifdef DBG2_GCTB
      cout << " candidates not exceeding the seed size; skipping " <<  endl;
      #endif
      continue;
    }
    //
    // Refit - keep existing trajectory in case fit is not possible
    // or fails
    //
    backwardFit(*it,nSeed,fitter,reFitted,seedHits);
    if ( reFitted.size()!=1 ) {
      rebuiltTrajectories.push_back(*it);
      //std::cout << "after reFitted.size() " << reFitted.size() << std::endl;
      continue;
    }
    //std::cout << "after reFitted.size() " << reFitted.size() << std::endl;
    //
    // Rebuild seeding part. In case it fails: keep initial trajectory
    // (better to drop it??)
    //
    int nRebuilt =
      rebuildSeedingRegion (seedHits,reFitted.front(),rebuiltTrajectories);
    if ( nRebuilt==0 )  rebuiltTrajectories.push_back(*it);
  }
  //
  // Replace input trajectories with new ones
  //
  result.swap(rebuiltTrajectories);
}

int
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion 
(const std::vector<const TrackingRecHit*>& seedHits, TempTrajectory& candidate,
 TrajectoryContainer& result) const 
{
  //
  // Try to rebuild one candidate in the seeding region.
  // The resulting trajectories are returned in result,
  // the count is the return value.
  //
  TrajectoryContainer rebuiltTrajectories;
#ifdef DBG2_GCTB
/*  const LayerFinderByDet layerFinder;
  if ( !seedHits.empty() && seedHits.front().isValid() ) {
    DetLayer* seedLayer = layerFinder(seedHits.front().det());
    cout << "Seed hit at " << seedHits.front().globalPosition()
	 << " " << seedLayer << endl;
    cout << "Started from " 
	 << candidate.lastMeasurement().updatedState().globalPosition().perp() << " "
	 << candidate.lastMeasurement().updatedState().globalPosition().z() << endl;
    pair<bool,TrajectoryStateOnSurface> layerComp(false,TrajectoryStateOnSurface());
    if ( seedLayer ) layerComp =
      seedLayer->compatible(candidate.lastMeasurement().updatedState(),
			      propagator(),estimator());
    pair<bool,TrajectoryStateOnSurface> detComp =
      seedHits.front().det().compatible(candidate.lastMeasurement().updatedState(),
					propagator(),estimator());
    cout << "  layer compatibility = " << layerComp.first;
    cout << "  det compatibility = " << detComp.first;
    if ( detComp.first ) {
      cout << "  estimate = " 
	   << estimator().estimate(detComp.second,seedHits.front()).second ;
    }
    cout << endl;
  }*/
  cout << "Before backward building: #measurements = " 
       << candidate.measurements().size() ; //<< endl;;
#endif
  //
  // Use standard building with standard cuts. Maybe better to use different
  // cuts from "forward" building (e.g. no check on nr. of invalid hits)?
  //
  groupedLimitedCandidates(candidate,(const TrajectoryFilter*)0, theBackwardPropagator, rebuiltTrajectories);
#ifdef DBG2_GCTB
  cout << "   After backward building: #measurements =";
  for ( TrajectoryContainer::iterator it=rebuiltTrajectories.begin();
	it!=rebuiltTrajectories.end(); it++ )  cout << " " << it->measurements().size();
  cout << endl;
#endif
  //
  // Check & count resulting candidates
  //
  int nrOfTrajectories(0);
  //const RecHitEqualByChannels recHitEqual(false,false);
  //vector<TM> oldMeasurements(candidate.measurements());
  for ( TrajectoryContainer::iterator it=rebuiltTrajectories.begin();
	it!=rebuiltTrajectories.end(); it++ ) {

    vector<TM> newMeasurements(it->measurements());
    //
    // Verify presence of seeding hits?
    //
    if ( theRequireSeedHitsInRebuild ) {
      // no hits found (and possibly some invalid hits discarded): drop track
      if ( newMeasurements.size()<=candidate.measurements().size() ){  
#ifdef DBG2_GCTB
	cout << "newMeasurements.size()<=candidate.measurements().size()" << endl;
#endif
	continue;
      }	
      // verify presence of hits
      if ( !verifyHits(newMeasurements.begin()+candidate.measurements().size(),
		       newMeasurements.end(),seedHits) ){
#ifdef DBG2_GCTB
	  cout << "seed hits not found in rebuild" << endl;	
#endif	
	  continue; 
      }
    }
    //
    // construct final trajectory in the right order
    //
    Trajectory reversedTrajectory(it->seed(),it->seed().direction());
    for ( vector<TM>::reverse_iterator im=newMeasurements.rbegin();
	  im!=newMeasurements.rend(); im++ ) {
      reversedTrajectory.push(*im);
    }
    // save & count result
    result.push_back(reversedTrajectory);
    nrOfTrajectories++;
#ifdef DBG2_GCTB
    cout << "New traj direction = " << reversedTrajectory.direction() << endl;
    vector<TM> tms = reversedTrajectory.measurements();
    for ( vector<TM>::const_iterator im=tms.begin();
	  im!=tms.end(); im++ ) {
      if ( im->recHit()->isValid() )  cout << im->recHit()->globalPosition();
      else cout << "(-,-,-)";
      cout << " ";
      cout << " fwdPred " << im->forwardPredictedState().isValid();
      cout << " bwdPred " << im->backwardPredictedState().isValid();
      cout << " upPred " << im->updatedState().isValid();
      //SimIdPrinter()(im->recHit());
      cout << endl;
    }
#endif
  }
  return nrOfTrajectories;
}

void
GroupedCkfTrajectoryBuilder::backwardFit (Trajectory& candidate, unsigned int nSeed,
						    const TrajectoryFitter& fitter,
						    TempTrajectoryContainer& fittedTracks,
						    std::vector<const TrackingRecHit*>& remainingHits) const
{
  //
  // clear array of non-fitted hits
  //
  remainingHits.clear();
  //
  // skip candidates which are not exceeding the seed size
  // (should not happen)
  //
  if ( candidate.measurements().size()<=nSeed ) {
    fittedTracks.clear();
    return;
  }
#ifdef DBG2_GCTB
    {
      cout << "nSeed " << nSeed << endl;
      cout << "Old traj direction = " << candidate.direction() << endl;
      vector<TM> tms = candidate.measurements();
      for ( vector<TM>::const_iterator im=tms.begin();
	    im!=tms.end(); im++ ) {
	if ( im->recHit()->isValid() )  cout << im->recHit()->globalPosition();
	else cout << "(-,-,-)";
	cout << " layer " << im->layer();
	cout << " fwdPred " << im->forwardPredictedState().isValid();
	cout << " bwdPred " << im->backwardPredictedState().isValid();
	cout << " upPred " << im->updatedState().isValid() << " ;";
	//SimIdPrinter()(im->recHit());
	cout << endl;
      }
    }
#endif
  //
  // backward fit trajectory (excluding the seeding region)
  //
  vector<TM> oldMeasurements(candidate.measurements());
//   int nOld(oldMeasurements.size());
//   const unsigned int nHitAllMin(5);
//   const unsigned int nHit2dMin(2);
  unsigned int nHit(0);    // number of valid hits after seeding region
  //unsigned int nHit2d(0);  // number of valid hits after seeding region with 2D info
  // use all hits except the first n (from seed), but require minimum
  // specified in configuration.
  //unsigned int nHitMin = max(oldMeasurements.size()-nSeed,theMinNrOfHitsForRebuild);
  unsigned int nHitMin = oldMeasurements.size()-nSeed;
  // we want to rebuild only if the number of measurements excluding the seed measurements is higher than the cut
  if (nHitMin<theMinNrOfHitsForRebuild){
	fittedTracks.clear();
    	return;
  }
  //cout << "nHitMin " << nHitMin << endl;
#ifdef DBG2_GCTB
  cout << "Sizes: " << oldMeasurements.size() << " / " << endl;
#endif
  //
  // create input trajectory for backward fit
  //
  Trajectory fwdTraj(candidate.seed(),oppositeDirection(candidate.direction()));
  //const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(), TrajectorySeed::recHitContainer(), oppositeDirection(candidate.direction()));
  //Trajectory fwdTraj(seed, oppositeDirection(candidate.direction()));
  std::vector<const DetLayer*> bwdDetLayer; 
  for ( vector<TM>::reverse_iterator im=oldMeasurements.rbegin();
	im!=oldMeasurements.rend(); im++ ) {
    const TrackingRecHit* hit = im->recHit()->hit();
    //
    // add hits until required number is reached
    //
    if ( nHit<nHitMin ){//|| nHit2d<theMinNrOf2dHitsForRebuild ) {
      fwdTraj.push(*im);
      bwdDetLayer.push_back(im->layer());
      //
      // count valid / 2D hits
      //
      if ( hit->isValid() ) {
	nHit++;
	//if ( hit.isMatched() ||
	//     hit.det().detUnits().front()->type().module()==pixel )
        //nHit2d++;
      }
    }
    //if (nHit==nHitMin) lastBwdDetLayer=im->layer();	
    //
    // keep remaining (valid) hits for verification
    //
    else if ( hit->isValid() ) {
      //std::cout << "Adding a remaining hit" << std::endl;
      remainingHits.push_back(hit);
    }
  }
  //
  // Fit only if required number of valid hits can be used
  //
  if ( nHit<nHitMin ){  //|| nHit2d<theMinNrOf2dHitsForRebuild ) {
    fittedTracks.clear();
    return;
  }
  //
  // Do the backward fit (important: start from scaled, not random cov. matrix!)
  //
  TrajectoryStateOnSurface firstTsos(fwdTraj.firstMeasurement().updatedState());
  //cout << "firstTsos "<< firstTsos << endl;
  firstTsos.rescaleError(100.);
  //TrajectoryContainer bwdFitted(fitter.fit(fwdTraj.seed(),fwdTraj.recHits(),firstTsos));
  TrajectoryContainer bwdFitted(fitter.fit(
  		TrajectorySeed(PTrajectoryStateOnDet(), TrajectorySeed::recHitContainer(), oppositeDirection(candidate.direction())),
  		fwdTraj.recHits(),firstTsos));
  if (bwdFitted.size()){
#ifdef DBG2_GCTB
  	cout << "Obtained " << bwdFitted.size() << " bwdFitted trajectories with measurement size " << bwdFitted.front().measurements().size() << endl;
#endif
	TempTrajectory fitted(fwdTraj.seed(), fwdTraj.direction());
        vector<TM> tmsbf = bwdFitted.front().measurements();
	int iDetLayer=0;
	//this is ugly but the TM in the fitted track do not contain the DetLayer.
	//So we have to cache the detLayer pointers and replug them in.
	//For the backward building it would be enaugh to cache the last DetLayer, 
	//but for the intermediary cleaning we need all
 	for ( vector<TM>::const_iterator im=tmsbf.begin();im!=tmsbf.end(); im++ ) {
		fitted.push(TM( (*im).forwardPredictedState(),
				(*im).backwardPredictedState(),
				(*im).updatedState(),
				(*im).recHit(),
				(*im).estimate(),
				bwdDetLayer[iDetLayer]));
#ifdef DBG2_GCTB
		if ( im->recHit()->isValid() )  cout << im->recHit()->globalPosition();
     		else cout << "(-,-,-)";
     		cout << " layer " << bwdDetLayer[iDetLayer];
     		cout << " fwdPred " << im->forwardPredictedState().isValid();
     		cout << " bwdPred " << im->backwardPredictedState().isValid();
     		cout << " upPred " << im->updatedState().isValid() << " ;";
     		//SimIdPrinter()(im->recHit());
     		cout << endl;	
#endif		
		iDetLayer++;
	}
/*
	TM lastMeas = bwdFitted.front().lastMeasurement();
	fitted.pop();
	fitted.push(TM(lastMeas.forwardPredictedState(), 
			       lastMeas.backwardPredictedState(), 
			       lastMeas.updatedState(),
			       lastMeas.recHit(),
			       lastMeas.estimate(),
                               lastBwdDetLayer));*/
	fittedTracks.push_back(fitted);
  }
  //
  // save result
  //
  //fittedTracks.swap(bwdFitted);
  //cout << "Obtained " << fittedTracks.size() << " fittedTracks trajectories with measurement size " << fittedTracks.front().measurements().size() << endl;
}

bool
GroupedCkfTrajectoryBuilder::verifyHits (vector<TM>::const_iterator tmBegin,
				         vector<TM>::const_iterator tmEnd,
					 const std::vector<const TrackingRecHit*>& hits) const
{
  //
  // verify presence of the seeding hits
  //
#ifdef DBG2_GCTB
  cout << "Checking for " << hits.size() << " hits in "
       << tmEnd-tmBegin << " measurements" << endl;
#endif
  for ( vector<const TrackingRecHit*>::const_iterator ir=hits.begin();
	ir!=hits.end(); ir++ ) {
    // assume that all seeding hits are valid!
    bool foundHit(false);
    for ( vector<TM>::const_iterator im=tmBegin; im!=tmEnd; im++ ) {
      if ( im->recHit()->isValid() && (*ir)->sharesInput(im->recHit()->hit(), TrackingRecHit::some) ) {
	foundHit = true;
	break;
      }
    }
    if ( !foundHit )  return false;
  }
  return true;
}




