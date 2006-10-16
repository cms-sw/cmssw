#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/TrajectorySegmentBuilder.h"


#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

//B.M. #include "CommonDet/DetLayout/interface/NavigationSetter.h"
//B.M. #include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
//B.M. #include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
//B.M. #include "CommonReco/TrackFitters/interface/KFTrajectoryFitter.h"
#include "RecoTracker/CkfPattern/interface/GroupedTrajCandLess.h"
//B.M. #include "CommonDet/BasicDet/interface/RecHitEqualByChannels.h"
#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/MaxHitsTrajectoryFilter.h"
#include "RecoTracker/CkfPattern/interface/RegionalTrajectoryFilter.h"
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

//B.M. #include "CommonDet/BasicDet/interface/DetType.h"

// only included for RecHit comparison operator:
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include <algorithm> 

//#define DBG_GCTB


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
			    const TransientTrackingRecHitBuilder* RecHitBuilder,
			    const MeasurementTracker*             measurementTracker):

  theUpdator(updator),thePropagatorAlong(propagatorAlong),
  thePropagatorOpposite(propagatorOpposite),theEstimator(estimator),
  theTTRHBuilder(RecHitBuilder),theMeasurementTracker(measurementTracker),
  theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
  theForwardPropagator(0),theBackwardPropagator(0),
  theMinPtCondition(new MinPtTrajectoryFilter(conf.getParameter<double>("ptCut"))),
  theMaxHitsCondition(new MaxHitsTrajectoryFilter(conf.getParameter<int>("maxNumberOfHits")))
{
  // fill data members from parameters (eventually data members could be dropped)
  //
  theMaxCand                  = conf.getParameter<int>("maxCand");
  theMaxLostHit               = conf.getParameter<int>("maxLostHit");
  theMaxConsecLostHit         = conf.getParameter<int>("maxConsecLostHit");
  theLostHitPenalty           = conf.getParameter<double>("lostHitPenalty");
  theFoundHitBonus            = conf.getParameter<double>("foundHitBonus");
  theIntermediateCleaning     = conf.getParameter<bool>("intermediateCleaning");
  theMinHits                  = conf.getParameter<int>("minimumNumberOfHits");
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

GroupedCkfTrajectoryBuilder::~GroupedCkfTrajectoryBuilder()
{
  //B.M. delete theConfigurableCondition;
  delete theMinPtCondition;
  delete theMaxHitsCondition;
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

  Trajectory startingTraj = createStartingTrajectory( seed);

  groupedLimitedCandidates( startingTraj, regionalCondition, result);
  if ( result.empty() )  return result;


  /* ========= B.M. to be ported later ====================
  //
  // try to additional hits in the seeding region
  //
  if ( theMinNrOfHitsForRebuild>0 ) {
    // reverse direction
    thePropagator->setPropagationDirection(oppositeDirection(seed.direction()));
    // rebuild part of the trajectory
    rebuildSeedingRegion(startingTraj,result);
  }
  =================================== */
  analyseResult(result);

#ifdef DBG_GCTB
  cout << "GroupedCkfTrajectoryBuilder: returning result of size " << result.size() << endl;
#endif

  return result;
}

Trajectory 
GroupedCkfTrajectoryBuilder::createStartingTrajectory( const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction());
  if (  seed.direction() == alongMomentum) {
    theForwardPropagator = &(*thePropagatorAlong);
    theBackwardPropagator = &(*thePropagatorOpposite);
  }
  else {
    theForwardPropagator = &(*thePropagatorOpposite);
    theBackwardPropagator = &(*thePropagatorAlong);
  }

  vector<TM> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
    }
  }
  return result;
}
  
bool GroupedCkfTrajectoryBuilder::qualityFilter( const Trajectory& traj) const
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

bool 
GroupedCkfTrajectoryBuilder::toBeContinued (const Trajectory& traj,
					    const TrajectoryFilter* regionalCondition) const
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
    else if ( // FIXME: !Trajectory::inactive(itm->recHit()->det()) &&
	     Trajectory::lost(*itm->recHit())) consecLostHit++;
  }
  if (consecLostHit > theMaxConsecLostHit) return false; 

  // stopping condition from region has highest priority
  //FIXME,restore this: if ( regionalCondition && !(*regionalCondition)(traj) )  return false;
  // next: pt-cut
  if ( !(*theMinPtCondition)(traj) )  return false;
  if ( !(*theMaxHitsCondition)(traj) )  return false;
  // finally: configurable condition
  //FIXME,restore this: if ( !(*theConfigurableCondition)(traj) )  return false;

  return true;
}

void 
GroupedCkfTrajectoryBuilder::addToResult (Trajectory& traj, 
					  TrajectoryContainer& result) const
{
  // quality check
  if ( !qualityFilter(traj) )  return;
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  result.push_back( traj);
}


void 
GroupedCkfTrajectoryBuilder::groupedLimitedCandidates (Trajectory& startingTraj, 
						       const TrajectoryFilter* regionalCondition, 
						       TrajectoryContainer& result) const
{
  TrajectoryContainer candidates;
  TrajectoryContainer newCand;
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      if ( !advanceOneLayer(*traj,regionalCondition,newCand,result) ) {
#ifdef DBG_GCTB
	cout << "GCTB: terminating after advanceOneLayer==false" << endl;
#endif
 	continue;
      }
    
#ifdef DBG_GCTB
      cout << "newCand(1)";
      for ( TrajectoryContainer::const_iterator it=newCand.begin();
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
      for ( TrajectoryContainer::const_iterator it=newCand.begin();
	    it!=newCand.end(); it++ ) 
	cout << " " << it->lostHits() << " " << it->foundHits() 
	     << " " << it->chiSquared() << " ;";
      cout << endl;
#endif
    }

#ifdef DBG_GCTB
    cout << "newCand.size() at end = " << newCand.size() << endl;
#endif

    /* ========= B.M. to be ported later =========
    if (theIntermediateCleaning) {
      candidates.clear();
      candidates = groupedIntermediaryClean(newCand);
    } else {
      candidates.swap(newCand);
    }
    ================ */
    candidates.swap(newCand);

#ifdef DBG_GCTB
    cout << "candidates(3)";
    for ( TrajectoryContainer::const_iterator it=candidates.begin();
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
GroupedCkfTrajectoryBuilder::advanceOneLayer (Trajectory& traj, 
					      const TrajectoryFilter* regionalCondition, 
					      TrajectoryContainer& newCand, 
					      TrajectoryContainer& result) const
{
  TSOS currentState(traj.lastMeasurement().updatedState());

  if ( traj.lastLayer()==0 ) {
    cout << "traj.lastLayer()==0; "
	 << "lastmeas pos r / z = " << currentState.globalPosition().perp() << " "
	 << currentState.globalPosition().z() << endl;
    return false;
  }
  vector<const DetLayer*> nl = 
    traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
  //   if (nl.empty()) {
  //     addToResult(traj,result);
  //     return false;
  //   }
  
#ifdef DBG_GCTB
  //B.M. TkLayerName layerName;
  //B.M. cout << "Started from " << layerName(traj.lastLayer()) 
  cout << "Started from "
       << " " << traj.lastLayer() << endl;
  cout << "Trying to go to";
  for ( vector<const DetLayer*>::iterator il=nl.begin(); 
	il!=nl.end(); il++) 
    //B.M. cout << " " << layerName(*il)  << " " << *il << endl;    
    cout << " " << *il << endl;    
  cout << endl;
#endif

  bool foundSegments(false);
  bool foundNewCandidates(false);
  for ( vector<const DetLayer*>::iterator il=nl.begin(); 
	il!=nl.end(); il++) {
    TrajectorySegmentBuilder layerBuilder(theMeasurementTracker,
					  theLayerMeasurements,
					  **il,*theForwardPropagator,
					  *theUpdator,*theEstimator,
					  theLockHits,theBestHitOnly);
    
#ifdef DBG_GCTB
    cout << "GCTB: starting from r / z = " << currentState.globalPosition().perp()
	 << " / " << currentState.globalPosition().z() << " , pt / pz = " 
	 << currentState.globalMomentum().perp() << " / " 
	 << currentState.globalMomentum().z() << " for layer at "
	 << *il << endl;
    cout << "     errors:";
    for ( int i=0; i<5; i++ )  cout << " " << sqrt(currentState.curvilinearError().matrix()[i][i]);
    cout << endl;
#endif

    vector<Trajectory> segments =
      layerBuilder.segments(traj.lastMeasurement().updatedState());

#ifdef DBG_GCTB
    cout << "GCTB: number of segments = " << segments.size() << endl;
#endif

    if ( !segments.empty() )  foundSegments = true;

    for ( vector<Trajectory>::const_iterator is=segments.begin();
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
      Trajectory newTraj(traj);
      for ( vector<TM>::const_iterator im=measurements.begin();
	    im!=measurements.end(); im++ )  newTraj.push(*im);
      if ( toBeContinued(newTraj,regionalCondition) ) {

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

/* ================= to be ported later ============================
GroupedCkfTrajectoryBuilder::TrajectoryContainer
GroupedCkfTrajectoryBuilder::groupedIntermediaryClean (TrajectoryContainer& theTrajectories)
{
  if (theTrajectories.empty()) return TrajectoryContainer();
  TrajectoryContainer result;
  
  RecHitEqualByChannels recHitEqualByChannels(false, false);
  int firstLayerSize, secondLayerSize;

  for (TrajectoryContainer::iterator firstTraj=theTrajectories.begin();
       firstTraj!=(theTrajectories.end()-1); firstTraj++) {

    if ( (!firstTraj->isValid()) ||
         (!firstTraj->lastMeasurement().recHit()->isValid()) ) continue;
    vector<TM> firstMeasurements(firstTraj->measurements());
    vector<const DetLayer*> firstLayers = layers(firstMeasurements);
    firstLayerSize = firstLayers.size();
    if ( firstLayerSize<4 )  continue;

    for (TrajectoryContainer::iterator secondTraj=(firstTraj+1);
       secondTraj!=theTrajectories.end(); secondTraj++) {

      if ( (!secondTraj->isValid()) ||
           (!secondTraj->lastMeasurement().recHit()->isValid()) ) continue;
      vector<TM> secondMeasurements(secondTraj->measurements());
      vector<const DetLayer*> secondLayers = layers(secondMeasurements);
      secondLayerSize = secondLayers.size();
      //
      // only candidates using the same last 3 layers are compared
      //
      if ( firstLayerSize!=secondLayerSize )  continue;
      if ( firstLayers[firstLayerSize-1]!=secondLayers[firstLayerSize-1] ||
	   firstLayers[firstLayerSize-2]!=secondLayers[firstLayerSize-2] ||
	   firstLayers[firstLayerSize-3]!=secondLayers[firstLayerSize-3] )  continue;

      vector<TM>::reverse_iterator im1 = firstMeasurements.rbegin();
      vector<TM>::reverse_iterator im2 = secondMeasurements.rbegin();
      //
      // check for identical hits in the last layer
      //
      bool unequal(false);
      const DetLayer* layerPtr = firstLayers[firstLayerSize-1];
      while ( im1!=firstMeasurements.rend()&&im2!=secondMeasurements.rend() ) {
	if ( im1->layer()!=layerPtr || im2->layer()!=layerPtr )  break;
	if ( !(im1->recHit()->isValid()) || !(im2->recHit()->isValid()) ||
	     !recHitEqualByChannels(im1->recHit(),im2->recHit()) ) {
	  unequal = true;
	  break;
	}
	im1++;
	im2++;
      }
      if ( im1==firstMeasurements.rend() || im2==secondMeasurements.rend() ||
	   im1->layer()==layerPtr || im2->layer()==layerPtr || unequal )  continue;
      //
      // check for invalid hits in the layer -2
      // compare only candidates with invalid / valid combination
      //
      layerPtr = firstLayers[firstLayerSize-2];
      bool firstValid(true);
      while ( im1!=firstMeasurements.rend()&&im1->layer()==layerPtr ) {
	if ( !im1->recHit()->isValid() )  firstValid = false;
	im1++;
      }
      bool secondValid(true);
      while ( im2!=secondMeasurements.rend()&&im2->layer()==layerPtr ) {
	if ( !im2->recHit()->isValid() )  secondValid = false;
	im2++;
      }
      if ( !tkxor(firstValid,secondValid) )  continue;
      //
      // ask for identical hits in layer -3
      //
      unequal = false;
      layerPtr = firstLayers[firstLayerSize-1];
      while ( im1!=firstMeasurements.rend()&&im2!=secondMeasurements.rend() ) {
	if ( im1->layer()!=layerPtr || im2->layer()!=layerPtr )  break;
	if ( !(im1->recHit()->isValid()) || !(im2->recHit()->isValid()) ||
	     !recHitEqualByChannels(im1->recHit(),im2->recHit()) ) {
	  unequal = true;
	  break;
	}
	im1++;
	im2++;
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

  for (TrajectoryBuilder::TrajectoryIterator it = theTrajectories.begin();
       it != theTrajectories.end(); it++) {
    if(it->isValid()) result.push_back( *it);
  }

  return result;
}

vector<const DetLayer*>
GroupedCkfTrajectoryBuilder::layers (const vector<TM>& measurements) const 
{
  vector<const DetLayer*> result;
  if ( measurements.empty() )  return result;

  result.push_back(measurements.front().layer());
  for ( vector<TM>::const_iterator im=measurements.begin()+1;
	im!=measurements.end(); im++ ) {
    if ( im->layer()!=result.back() )  result.push_back(im->layer());
  }

  return result;
}

void
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion 
(Trajectory& startingTraj, TrajectoryContainer& result)
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
  KFTrajectoryFitter fitter(*thePropagator,updator(),estimator());
  //
  vector<Trajectory> reFitted;
  vector<RecHit> seedHits(startingTraj.seed().recHits());
  unsigned int nSeed(seedHits.size());
  TrajectoryContainer rebuiltTrajectories;
  for ( TrajectoryContainer::iterator it=result.begin();
	it!=result.end(); it++ ) {
    //
    // skip candidates which are not exceeding the seed size
    // (should not happen) - keep existing trajectory
    //
    if ( it->measurements().size()<=startingTraj.measurements().size() ) {
      rebuiltTrajectories.push_back(*it);
      continue;
    }
    //
    // Refit - keep existing trajectory in case fit is not possible
    // or fails
    //
    backwardFit(*it,nSeed,fitter,reFitted,seedHits);
    if ( reFitted.size()!=1 ) {
      rebuiltTrajectories.push_back(*it);
      continue;
    }
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
(const vector<RecHit>& seedHits, Trajectory& candidate,
 TrajectoryContainer& result)
{
  //
  // Try to rebuild one candidate in the seeding region.
  // The resulting trajectories are returned in result,
  // the count is the return value.
  //
  TrajectoryContainer rebuiltTrajectories;
#ifdef DBG2_GCTB
  const LayerFinderByDet layerFinder;
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
  }
  cout << "Before backward building: #measurements = " 
       << candidate.measurements().size() << endl;;
#endif
  //
  // Use standard building with standard cuts. Maybe better to use different
  // cuts from "forward" building (e.g. no check on nr. of invalid hits)?
  //
  groupedLimitedCandidates(candidate,(const TrajectoryFilter*)0,rebuiltTrajectories);
#ifdef DBG2_GCTB
  cout << "After backward building: #measurements =";
  for ( TrajectoryContainer::iterator it=rebuiltTrajectories.begin();
	it!=rebuiltTrajectories.end(); it++ )  cout << " " << it->measurements().size();
  cout << endl;
#endif
  //
  // Check & count resulting candidates
  //
  int nrOfTrajectories(0);
  const RecHitEqualByChannels recHitEqual(false,false);
  vector<TM> oldMeasurements(candidate.measurements());
  for ( TrajectoryContainer::iterator it=rebuiltTrajectories.begin();
	it!=rebuiltTrajectories.end(); it++ ) {

    vector<TM> newMeasurements(it->measurements());
    //
    // Verify presence of seeding hits?
    //
    if ( theRequireSeedHitsInRebuild ) {
      // no hits found (and possibly some invalid hits discarded): drop track
      if ( newMeasurements.size()<=oldMeasurements.size() )  continue;
      // verify presence of hits
      if ( !verifyHits(newMeasurements.begin()+oldMeasurements.size(),
		       newMeasurements.end(),recHitEqual,seedHits) )  continue;
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
      SimIdPrinter()(im->recHit());
      cout << endl;
    }
#endif
  }
  return nrOfTrajectories;
}

void
GroupedCkfTrajectoryBuilder::backwardFit (Trajectory& candidate, unsigned int nSeed,
						    const TrajectoryFitter& fitter,
						    TrajectoryContainer& fittedTracks,
						    vector<RecHit>& remainingHits) const
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
	SimIdPrinter()(im->recHit());
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
  unsigned int nHit2d(0);  // number of valid hits after seeding region with 2D info
  // use all hits except the first n (from seed), but require minimum
  // specified in configuration.
  unsigned int nHitMin = max(oldMeasurements.size()-nSeed,theMinNrOfHitsForRebuild);
#ifdef DBG2_GCTB
  cout << "Sizes: " << oldMeasurements.size() << " / " << nskip << endl;
#endif
  //
  // create input trajectory for backward fit
  //
  Trajectory fwdTraj(candidate.seed(),oppositeDirection(candidate.direction()));
  for ( vector<TM>::reverse_iterator im=oldMeasurements.rbegin();
	im!=oldMeasurements.rend(); im++ ) {
    RecHit hit(im->recHit());
    //
    // add hits until required number is reached
    //
    if ( nHit<nHitMin || nHit2d<theMinNrOf2dHitsForRebuild ) {
      fwdTraj.push(*im);
      //
      // count valid / 2D hits
      //
      if ( hit.isValid() ) {
	nHit++;
	if ( hit.isMatched() ||
	     hit.det().detUnits().front()->type().module()==pixel )
	  nHit2d++;
      }
    }
    //
    // keep remaining (valid) hits for verification
    //
    else if ( hit.isValid() ) {
      remainingHits.push_back(hit);
    }
  }
  //
  // Fit only if required number of hits can be used
  //
  if ( nHit<nHitMin || nHit2d<theMinNrOf2dHitsForRebuild ) {
    fittedTracks.clear();
    return;
  }
  //
  // Do the backward fit (important: start from scaled, not random cov. matrix!)
  //
  TrajectoryStateOnSurface firstTsos(fwdTraj.firstMeasurement().updatedState());
  firstTsos.rescaleError(100.);
  TrajectoryContainer bwdFitted(fitter.fit(fwdTraj.seed(),fwdTraj.recHits(),firstTsos));
#ifdef DBG2_GCTB
  cout << "Obtained " << bwdFitted.size() << " bwdFitted trajectories" << endl;
#endif
  //
  // save result
  //
  fittedTracks.swap(bwdFitted);
}

bool
GroupedCkfTrajectoryBuilder::verifyHits (vector<TM>::const_iterator tmBegin,
						   vector<TM>::const_iterator tmEnd,
						   const RecHitEqualByChannels& recHitEqual,
						   const vector<RecHit>& hits) const
{
  //
  // verify presence of the seeding hits
  //
#ifdef DBG2_GCTB
  cout << "Checking for " << hits.size() << " hits in "
       << tmEnd-tmBegin << " measurements" << endl;
#endif
  for ( vector<RecHit>::const_iterator ir=hits.begin();
	ir!=hits.end(); ir++ ) {
    // assume that all seeding hits are valid!
    bool foundHit(false);
    for ( vector<TM>::const_iterator im=tmBegin; im!=tmEnd; im++ ) {
      if ( im->recHit()->isValid() && recHitEqual(*ir,im->recHit()) ) {
	foundHit = true;
	break;
      }
    }
    if ( !foundHit )  return false;
  }
  return true;
}

================================== */


// method copied from CkfTrajectoryBuilder.cc
// it should be put in a common place for both algos

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

std::vector<TrajectoryMeasurement> 
GroupedCkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
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
	edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
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
  return result;
}
