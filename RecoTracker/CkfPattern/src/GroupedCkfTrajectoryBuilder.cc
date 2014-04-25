#include "RecoTracker/CkfPattern/interface/GroupedCkfTrajectoryBuilder.h"
#include "TrajectorySegmentBuilder.h"


#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "RecoTracker/CkfPattern/interface/GroupedTrajCandLess.h"
#include "TrackingTools/TrajectoryFiltering/interface/RegionalTrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
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
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"


// only included for RecHit comparison operator:
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

// for looper reconstruction
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include <algorithm>
#include <array>

namespace {
#ifdef STAT_TSB
  struct StatCount {
    long long totSeed;
    long long totTraj;
    long long totRebuilt;
    long long totInvCand;
    void zero() {
      totSeed=totTraj=totRebuilt=totInvCand=0;
    }
    void traj(long long t) {
      totTraj+=t;
    }
    void seed() {++totSeed;}
    void rebuilt(long long t) {
      totRebuilt+=t;
    }
    void invalid() { ++totInvCand;}
    void print() const {
      std::cout << "GroupedCkfTrajectoryBuilder stat\nSeed/Traj/Rebuilt "
    		<<  totSeed <<'/'<< totTraj <<'/'<< totRebuilt
		<< std::endl;
    }
    StatCount() { zero();}
    ~StatCount() { print();}
  };
  StatCount statCount;

#else
  struct StatCount {
    void traj(long long){}
    void seed() {}
    void rebuilt(long long) {}
    void invalid() {}
  };
  [[cms::thread_safe]] StatCount statCount;
#endif


}



using namespace std;

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

GroupedCkfTrajectoryBuilder::GroupedCkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector& iC):
  BaseCkfTrajectoryBuilder(conf,
                           BaseCkfTrajectoryBuilder::createTrajectoryFilter(conf.getParameter<edm::ParameterSet>("trajectoryFilter"), iC),
                           conf.getParameter<bool>("useSameTrajFilter") ?
                             BaseCkfTrajectoryBuilder::createTrajectoryFilter(conf.getParameter<edm::ParameterSet>("trajectoryFilter"), iC) :
                             BaseCkfTrajectoryBuilder::createTrajectoryFilter(conf.getParameter<edm::ParameterSet>("inOutTrajectoryFilter"), iC)
                           )
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
  theKeepOriginalIfRebuildFails = conf.getParameter<bool>("keepOriginalIfRebuildFails");
  theMinNrOfHitsForRebuild    = max(0,conf.getParameter<int>("minNrOfHitsForRebuild"));
  maxPt2ForLooperReconstruction     = conf.existsAs<double>("maxPtForLooperReconstruction") ? 
    conf.getParameter<double>("maxPtForLooperReconstruction") : 0;
  maxPt2ForLooperReconstruction *=maxPt2ForLooperReconstruction;
  maxDPhiForLooperReconstruction     = conf.existsAs<double>("maxDPhiForLooperReconstruction") ? 
    conf.getParameter<double>("maxDPhiForLooperReconstruction") : 2.0;


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

/*
  void GroupedCkfTrajectoryBuilder::setEvent(const edm::Event& event) const
  {
  theMeasurementTracker->update(event);
}
*/

void GroupedCkfTrajectoryBuilder::setEvent_(const edm::Event& event, const edm::EventSetup& iSetup) {
}

GroupedCkfTrajectoryBuilder::TrajectoryContainer 
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed) const 
{
  TrajectoryContainer ret; 
  ret.reserve(10);
  buildTrajectories(seed, ret, 0);
  return ret; 
}

GroupedCkfTrajectoryBuilder::TrajectoryContainer 
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed, 
					   const TrackingRegion& region) const
{
  TrajectoryContainer ret; 
  ret.reserve(10);
  RegionalTrajectoryFilter regionalCondition(region);
  buildTrajectories(seed, ret, &regionalCondition);
  return ret; 
}

void 
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed, GroupedCkfTrajectoryBuilder::TrajectoryContainer &ret) const 
{
  buildTrajectories(seed,ret,0);
}

void
GroupedCkfTrajectoryBuilder::trajectories (const TrajectorySeed& seed, 
                                            GroupedCkfTrajectoryBuilder::TrajectoryContainer &ret,
					    const TrackingRegion& region) const
{
  RegionalTrajectoryFilter regionalCondition(region);
  buildTrajectories(seed,ret,&regionalCondition);
}

void  
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion(const TrajectorySeed& seed,
						  TrajectoryContainer& result) const {    
  TempTrajectory const & startingTraj = createStartingTrajectory( seed);
  rebuildTrajectories(startingTraj,seed,result);

}

void
GroupedCkfTrajectoryBuilder::rebuildTrajectories(TempTrajectory const & startingTraj, const TrajectorySeed& seed,
						 TrajectoryContainer& result) const {
  TempTrajectoryContainer work;

  TrajectoryContainer final;

  work.reserve(result.size());
  for (TrajectoryContainer::iterator traj=result.begin();
       traj!=result.end(); ++traj) {
    if(traj->isValid()) work.push_back(TempTrajectory(std::move(*traj)));
  }

  rebuildSeedingRegion(seed,startingTraj,work);
  final.reserve(work.size());

  // better the seed to be always the same... 
  boost::shared_ptr<const TrajectorySeed>  sharedSeed;
  if (result.empty()) 
    sharedSeed.reset(new TrajectorySeed(seed));
   else sharedSeed = result.front().sharedSeed();


  for (TempTrajectoryContainer::iterator traj=work.begin();
       traj!=work.end(); ++traj) {
    final.push_back(traj->toTrajectory()); final.back().setSharedSeed(sharedSeed);
  }
  
  result.swap(final);

  statCount.rebuilt(result.size());

}

TempTrajectory
GroupedCkfTrajectoryBuilder::buildTrajectories (const TrajectorySeed& seed,
                                                GroupedCkfTrajectoryBuilder::TrajectoryContainer &result,
						const TrajectoryFilter* regionalCondition) const
{
  if (theMeasurementTracker == 0) {
      throw cms::Exception("LogicError") << "Asking to create trajectories to an un-initialized GroupedCkfTrajectoryBuilder.\nYou have to call clone(const MeasurementTrackerEvent *data) and then call trajectories on it instead.\n";
  }
 
  statCount.seed();
  //
  // Build trajectory outwards from seed
  //

  analyseSeed( seed);

  TempTrajectory const & startingTraj = createStartingTrajectory( seed);

  work_.clear();
  const bool inOut = true;
  groupedLimitedCandidates(seed, startingTraj, regionalCondition, forwardPropagator(seed), inOut, work_);
  if ( work_.empty() )  return startingTraj;



  /*  rebuilding is de-coupled from standard building
  //
  // try to additional hits in the seeding region
  //
  if ( theMinNrOfHitsForRebuild>0 ) {
    // reverse direction
    //thePropagator->setPropagationDirection(oppositeDirection(seed.direction()));
    // rebuild part of the trajectory
    rebuildSeedingRegion(startingTraj,work);
  }

  */
  boost::shared_ptr<const TrajectorySeed> pseed(new TrajectorySeed(seed));
  result.reserve(work_.size());
  for (TempTrajectoryContainer::const_iterator it = work_.begin(), ed = work_.end(); it != ed; ++it) {
    result.push_back( it->toTrajectory() ); result.back().setSharedSeed(pseed);
  }

  work_.clear(); 
  if (work_.capacity() > work_MaxSize_) {  TempTrajectoryContainer().swap(work_); work_.reserve(work_MaxSize_/2); }

  analyseResult(result);

  LogDebug("CkfPattern")<< "GroupedCkfTrajectoryBuilder: returning result of size " << result.size();
  statCount.traj(result.size());

#ifdef VI_DEBUG
  int kt=0;
  for (auto const & traj : result) {
int chit[7]={};
for (auto const & tm : traj.measurements()) {
  auto const & hit = tm.recHitR();
  if (!hit.isValid()) ++chit[0];
  if (hit.det()==nullptr) ++chit[1];
  if ( trackerHitRTTI::isUndef(hit) ) continue;
  if ( hit.dimension()!=2 ) {
    ++chit[2];
  } else {
    auto const & thit = static_cast<BaseTrackerRecHit const&>(hit);
    auto const & clus = thit.firstClusterRef();
    if (clus.isPixel()) ++chit[3];
    else if (thit.isMatched()) {
      ++chit[4];
    } else  if (thit.isProjected()) {
      ++chit[5];
    } else {
      ++chit[6];
        }
  }
 }

std::cout << "ckf " << kt++ << ": "; for (auto c:chit) std::cout << c <<'/'; std::cout<< std::endl;
}
#endif

  return startingTraj;

}


void 
GroupedCkfTrajectoryBuilder::groupedLimitedCandidates (const TrajectorySeed& seed,
                                                       TempTrajectory const& startingTraj, 
						       const TrajectoryFilter* regionalCondition,
						       const Propagator* propagator, 
                                                       bool inOut,
						       TempTrajectoryContainer& result) const
{
  unsigned int nIter=1;
  TempTrajectoryContainer candidates;
  TempTrajectoryContainer newCand;
  candidates.push_back( startingTraj);

  while ( !candidates.empty()) {

    newCand.clear();
    for (TempTrajectoryContainer::iterator traj=candidates.begin();
	 traj!=candidates.end(); traj++) {
      if ( !advanceOneLayer(seed, *traj, regionalCondition, propagator, inOut, newCand, result) ) {
	LogDebug("CkfPattern")<< "GCTB: terminating after advanceOneLayer==false";
 	continue;
      }

      LogDebug("CkfPattern")<<"newCand(1): after advanced one layer:\n"<<PrintoutHelper::dumpCandidates(newCand);

      if ((int)newCand.size() > theMaxCand) {
	//ShowCand()(newCand);

 	sort( newCand.begin(), newCand.end(), GroupedTrajCandLess(theLostHitPenalty,theFoundHitBonus));
 	newCand.erase( newCand.begin()+theMaxCand, newCand.end());
      }
      LogDebug("CkfPattern")<<"newCand(2): after removing extra candidates.\n"<<PrintoutHelper::dumpCandidates(newCand);
    }

    LogDebug("CkfPattern") << "newCand.size() at end = " << newCand.size();
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

    LogDebug("CkfPattern") <<"candidates(3): "<<result.size()<<" candidates after "<<nIter++<<" groupedCKF iteration: \n"
      			   <<PrintoutHelper::dumpCandidates(result)
			   <<"\n "<<candidates.size()<<" running candidates are: \n"
			   <<PrintoutHelper::dumpCandidates(candidates);
  }
}

#ifdef EDM_ML_DEBUG
std::string whatIsTheNextStep(TempTrajectory const& traj , std::pair<TrajectoryStateOnSurface,std::vector<const DetLayer*> >& stateAndLayers){
  std::stringstream buffer;
  vector<const DetLayer*> & nl = stateAndLayers.second;
  // #include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
  // #include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
  //B.M. TkLayerName layerName;
  //B.M. buffer << "Started from " << layerName(traj.lastLayer()) 
  const BarrelDetLayer* sbdl = dynamic_cast<const BarrelDetLayer*>(traj.lastLayer());
  const ForwardDetLayer* sfdl = dynamic_cast<const ForwardDetLayer*>(traj.lastLayer());
  if (sbdl) {
    buffer << "Started from " << traj.lastLayer() << " r=" << sbdl->specificSurface().radius() 
	   << " phi=" << sbdl->specificSurface().phi() << endl;
  } else if (sfdl) {
    buffer << "Started from " << traj.lastLayer() << " z " << sfdl->specificSurface().position().z()
	   << " phi " << sfdl->specificSurface().phi() << endl;
  }
  buffer << "Trying to go to";
  for ( vector<const DetLayer*>::iterator il=nl.begin();
	il!=nl.end(); il++){ 
    //B.M. buffer << " " << layerName(*il)  << " " << *il << endl;
    const BarrelDetLayer* bdl = dynamic_cast<const BarrelDetLayer*>(*il);
    const ForwardDetLayer* fdl = dynamic_cast<const ForwardDetLayer*>(*il);
    
    if (bdl) buffer << " r " << bdl->specificSurface().radius() << endl;
    if (fdl) buffer << " z " << fdl->specificSurface().position().z() << endl;
    //buffer << " " << *il << endl;   
  }
  return buffer.str();
}

std::string whatIsTheStateToUse(TrajectoryStateOnSurface &initial, TrajectoryStateOnSurface & stateToUse, const DetLayer * l){
  std::stringstream buffer;
  buffer << "GCTB: starting from " 
         << " r / phi / z = " << stateToUse.globalPosition().perp()
	 << " / " << stateToUse.globalPosition().phi()
	 << " / " << stateToUse.globalPosition().z() 
         << " , pt / phi / pz /charge = " 
	 << stateToUse.globalMomentum().perp() << " / "  
	 << stateToUse.globalMomentum().phi() << " / " 
    	 << stateToUse.globalMomentum().z() << " / " 
	 << stateToUse.charge()
	 << " for layer at "<< l << endl;
  buffer << "     errors:";
  for ( int i=0; i<5; i++ )  buffer << " " << sqrt(stateToUse.curvilinearError().matrix()(i,i));
  buffer << endl;
  
  //buffer << "GCTB: starting from r / phi / z = " << initial.globalPosition().perp()
  //<< " / " << initial.globalPosition().phi()
  //<< " / " << initial.globalPosition().z() << " , pt / pz = " 
  //<< initial.globalMomentum().perp() << " / " 
  //<< initial.globalMomentum().z() << " for layer at "
  //<< l << endl;
  //buffer << "     errors:";
  //for ( int i=0; i<5; i++ )  buffer << " " << sqrt(initial.curvilinearError().matrix()(i,i));
  //buffer << endl;
  return buffer.str();
}
#endif

bool 
GroupedCkfTrajectoryBuilder::advanceOneLayer (const TrajectorySeed& seed,
                                              TempTrajectory& traj, 
					      const TrajectoryFilter* regionalCondition, 
					      const Propagator* propagator,
                                              bool inOut,
					      TempTrajectoryContainer& newCand, 
					      TempTrajectoryContainer& result) const
{
  std::pair<TSOS,std::vector<const DetLayer*> > && stateAndLayers = findStateAndLayers(traj);


  if(maxPt2ForLooperReconstruction>0){
    if(
       //stateAndLayers.second.size()==0 &&
       traj.lastLayer()->location()==0) {
      float pt2 = stateAndLayers.first.globalMomentum().perp2();
      if (pt2<maxPt2ForLooperReconstruction &&
	  pt2>(0.3f*0.3f)
	  )
	stateAndLayers.second.push_back(traj.lastLayer());
    }
  }

  auto layerBegin = stateAndLayers.second.begin();
  auto layerEnd   = stateAndLayers.second.end();

  //   if (nl.empty()) {
  //     addToResult(traj,result,inOut);
  //     return false;
  //   }
  
#ifdef EDM_ML_DEBUG
  LogDebug("CkfPattern")<<whatIsTheNextStep(traj, stateAndLayers);
#endif
  
  bool foundSegments(false);
  bool foundNewCandidates(false);
  for ( auto il=layerBegin; il!=layerEnd; il++) {

    TSOS stateToUse = stateAndLayers.first;
    
    double dPhiCacheForLoopersReconstruction(0);
    if unlikely((*il)==traj.lastLayer()){
	
	if(maxPt2ForLooperReconstruction>0){
	  // ------ For loopers reconstruction
	  //cout<<" self propagating in advanceOneLayer (for loopers) \n";
	  const BarrelDetLayer* sbdl = dynamic_cast<const BarrelDetLayer*>(traj.lastLayer());
	  if(sbdl){
	    HelixBarrelCylinderCrossing cylinderCrossing(stateToUse.globalPosition(),
							 stateToUse.globalMomentum(),
							 stateToUse.transverseCurvature(),
							 propagator->propagationDirection(),
							 sbdl->specificSurface());
	    if(!cylinderCrossing.hasSolution()) continue;
	    GlobalPoint starting = stateToUse.globalPosition();
	    GlobalPoint target1 = cylinderCrossing.position1();
	    GlobalPoint target2 = cylinderCrossing.position2();
	    
	    GlobalPoint farther = fabs(starting.phi()-target1.phi()) > fabs(starting.phi()-target2.phi()) ?
	      target1 : target2;
	    
	    const Bounds& bounds( sbdl->specificSurface().bounds());
	    float length = 0.5f*bounds.length();
	    
	    /*
	      cout << "starting: " << starting << endl;
	      cout << "target1: " << target1 << endl;
	      cout << "target2: " << target2 << endl;
	      cout << "dphi: " << (target1.phi()-target2.phi()) << endl;
	    cout << "length: " << length << endl;
	    */
	    
	    /*
	      float deltaZ = bounds.thickness()/2.f/fabs(tan(stateToUse.globalDirection().theta()) ) ;
	      if(stateToUse.hasError())
	      deltaZ += 3*sqrt(stateToUse.cartesianError().position().czz());
	      if( fabs(farther.z()) > length + deltaZ ) continue;
	    */
	    if(fabs(farther.z())*0.95f>length) continue;
	    
	    Geom::Phi<float> tmpDphi = target1.phi()-target2.phi();
	    if(std::abs(tmpDphi)>maxDPhiForLooperReconstruction) continue;
	    GlobalPoint target(0.5f*(target1.basicVector()+target2.basicVector()));
	    //cout << "target: " << target << endl;
	    

	    
	    TransverseImpactPointExtrapolator extrapolator;
	    stateToUse = extrapolator.extrapolate(stateToUse, target, *propagator);
	    if (!stateToUse.isValid()) continue; //SK: consider trying the original? probably not
	    
	    //dPhiCacheForLoopersReconstruction = fabs(target1.phi()-target2.phi())*2.;
	    dPhiCacheForLoopersReconstruction = std::abs(tmpDphi);
	    traj.incrementLoops();
	  }else{ // not barrel
	    continue;
	  }
	}else{ // loopers not requested (why else???)
	// ------ For cosmics reconstruction
	  LogDebug("CkfPattern")<<" self propagating in advanceOneLayer.\n from: \n"<<stateToUse;
	  //self navigation case
	  // go to a middle point first
	  TransverseImpactPointExtrapolator middle;
	  GlobalPoint center(0,0,0);
	  stateToUse = middle.extrapolate(stateToUse, center, *(forwardPropagator(seed)));
	  
	  if (!stateToUse.isValid()) continue;
	  LogDebug("CkfPattern")<<"to: "<<stateToUse;
	}
      } // last layer... 
    
    //unsigned int maxCandidates = theMaxCand > 21 ? theMaxCand*2 : 42; //limit the number of returned segments
    LayerMeasurements layerMeasurements(theMeasurementTracker->measurementTracker(), *theMeasurementTracker);
    TrajectorySegmentBuilder layerBuilder(&layerMeasurements,
					  **il,*propagator,
					  *theUpdator,*theEstimator,
					  theLockHits,theBestHitOnly,theMaxCand);

#ifdef EDM_ML_DEBUG
    LogDebug("CkfPattern")<<whatIsTheStateToUse(stateAndLayers.first,stateToUse,*il);
#endif
    
    auto && segments= layerBuilder.segments(stateToUse);

    LogDebug("CkfPattern")<< "GCTB: number of segments = " << segments.size();

    if ( !segments.empty() )  foundSegments = true;
    
    for (auto is=segments.begin(); is!=segments.end(); is++ ) {
      //
      // assume "invalid hit only" segment is last in list
      //
      auto const & measurements = is->measurements();
      if ( !theAlwaysUseInvalid && is!=segments.begin() && measurements.size()==1 && 
	   (measurements.front().recHit()->getType() == TrackingRecHit::missing) )  break;
      

     //----  avoid to add the same hits more than once in the trajectory ----
      bool toBeRejected(false);
      for(auto revIt = measurements.rbegin(); revIt!=measurements.rend(); --revIt){
	// int tmpCounter(0);
	for(auto  newTrajMeasIt = traj.measurements().rbegin(); 
	    newTrajMeasIt != traj.measurements().rend(); --newTrajMeasIt){
	  //if(tmpCounter==2) break;
	  if(revIt->recHitR().geographicalId()==newTrajMeasIt->recHitR().geographicalId() 
	     && (revIt->recHitR().geographicalId() != DetId(0)) ){
	    toBeRejected=true;
	    goto rejected; //break;  // see http://stackoverflow.com/questions/1257744/can-i-use-break-to-exit-multiple-nested-for-loops
	  }
	  // tmpCounter++;
	}
      }
      
    rejected:;    // http://xkcd.com/292/
      if(toBeRejected){
#ifdef VI_DEBUG
        cout << "WARNING: neglect candidate because it contains the same hit twice \n";
          cout << "-- discarded track's pt,eta,#found/lost: "
          << traj.lastMeasurement().updatedState().globalMomentum().perp() << " , "
          << traj.lastMeasurement().updatedState().globalMomentum().eta() << " , "
          << traj.foundHits() << '/' << traj.lostHits() << "\n";
#endif
	traj.setDPhiCacheForLoopersReconstruction(dPhiCacheForLoopersReconstruction);
	continue; //Are we sure about this????
      }
      // ------------------------
      
      //
      // create new candidate
      //
      TempTrajectory newTraj(traj);
      traj.setDPhiCacheForLoopersReconstruction(dPhiCacheForLoopersReconstruction);
      newTraj.join(*is);

      //std::cout << "DEBUG: newTraj after push found,lost: " 
      //	  << newTraj.foundHits() << " , " 
      //	  << newTraj.lostHits() << " , "
      //	  << newTraj.measurements().size() << std::endl;
      
      
      
       //GIO// for ( vector<TM>::const_iterator im=measurements.begin();
      //GIO//        im!=measurements.end(); im++ )  newTraj.push(*im);
      //if ( toBeContinued(newTraj,regionalCondition) ) { TOBE FIXED
      if ( toBeContinued(newTraj, inOut) ) {
	// Have added one more hit to track candidate
	
	LogDebug("CkfPattern")<<"GCTB: adding updated trajectory to candidates: inOut="<<inOut<<" hits="<<newTraj.foundHits();

	newCand.push_back(std::move(newTraj));
	foundNewCandidates = true;
      }
      else {
	// Have finished building this track. Check if it passes cuts.

	LogDebug("CkfPattern")<< "GCTB: adding completed trajectory to results if passes cuts: inOut="<<inOut<<" hits="<<newTraj.foundHits();

	moveToResult(std::move(newTraj), result, inOut);
      }
    } // loop over segs
  } // loop over layers

  if ( !foundSegments ){
    LogDebug("CkfPattern")<< "GCTB: adding input trajectory to result";
    addToResult(traj, result, inOut);
  }
  return foundNewCandidates;
}

namespace {
/// fills in a list of layers from a container of TrajectoryMeasurements
/// 
  struct LayersInTraj {
    static constexpr int N=3;
    TempTrajectory * traj;
    std::array<DetLayer const *,N> layers;
    int tot;
    void fill(TempTrajectory & t) {
      traj = &t;
      tot=0;
      const TempTrajectory::DataContainer& measurements = traj->measurements();
      
      auto currl = layers[tot] = measurements.back().layer();
      TempTrajectory::DataContainer::const_iterator ifirst = measurements.rbegin();
      --ifirst;	 
      for ( TempTrajectory::DataContainer::const_iterator im=ifirst;
	    im!=measurements.rend(); --im ) {
	if ( im->layer()!=currl ) { ++tot; currl = im->layer(); if (tot<N)  layers[tot] = currl;}
      }
      ++tot;
    }
  
    //void verify() {
    //  for (vector<const DetLayer*>::const_iterator iter = result.begin(); iter != result.end(); iter++)
    //  if (!*iter) edm::LogWarning("CkfPattern")<< "Warning: null det layer!! ";
    // }
  };
}


//TempTrajectoryContainer
void
GroupedCkfTrajectoryBuilder::groupedIntermediaryClean (TempTrajectoryContainer& theTrajectories) const 
{
  //if (theTrajectories.empty()) return TrajectoryContainer();
  //TrajectoryContainer result;
  if (theTrajectories.empty()) return;  
  //RecHitEqualByChannels recHitEqualByChannels(false, false);
  LayersInTraj layers[theTrajectories.size()];
  int ntraj=0;
  for ( auto & t :  theTrajectories) {
    if ( t.isValid() && t.lastMeasurement().recHitR().isValid() )
      layers[ntraj++].fill(t);
  }

  if (ntraj<2) return;

  for (int ifirst=0; ifirst!=ntraj-1; ++ifirst) {
    auto firstTraj = layers[ifirst].traj;
    if (!firstTraj->isValid()) continue;
    const TempTrajectory::DataContainer & firstMeasurements = firstTraj->measurements();
    
    int firstLayerSize = layers[ifirst].tot;
    if ( firstLayerSize<4 )  continue;
    auto const & firstLayers = layers[ifirst].layers;

    for (int isecond= ifirst+1; isecond!=ntraj; ++isecond) {
      auto secondTraj = layers[isecond].traj;
      if (!secondTraj->isValid()) continue;

      const TempTrajectory::DataContainer & secondMeasurements = secondTraj->measurements();
      
      int secondLayerSize = layers[isecond].tot;
      //
      // only candidates using the same last 3 layers are compared
      //
      if ( firstLayerSize!=secondLayerSize )  continue;  // V.I.  why equal???
      auto const & secondLayers =  layers[isecond].layers;
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
	     !im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::some) ) {
	  //!im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::all) ) {
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
	     !im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::some) ) {
	  //!im1->recHit()->hit()->sharesInput(im2->recHit()->hit(), TrackingRecHit::all) ) {
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
	secondTraj->invalidate();   // V.I. why break?
	break;
      }
    } // second
  } // first 
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


void
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion(const TrajectorySeed&seed,
						  TempTrajectory const & startingTraj,
						  TempTrajectoryContainer& result) const
{
  //
  // Rebuilding of trajectories. Candidates are taken from result,
  // which will be replaced with the solutions after rebuild
  // (assume vector::swap is more efficient than building new container)
  //
  LogDebug("CkfPattern")<< "Starting to rebuild " << result.size() << " tracks";
  //
  // Fitter (need to create it here since the propagation direction
  // might change between different starting trajectories)
  //
  auto hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(hitBuilder())->cloner();
  KFTrajectoryFitter fitter(backwardPropagator(seed),&updator(),&estimator(),3,nullptr,&hitCloner);
  //
  TrajectorySeed::range rseedHits = seed.recHits();
  std::vector<const TrackingRecHit*> seedHits;
  //seedHits.insert(seedHits.end(), rseedHits.first, rseedHits.second);
  //for (TrajectorySeed::recHitContainer::const_iterator iter = rseedHits.first; iter != rseedHits.second; iter++){
  //	seedHits.push_back(&*iter);
  //}

  //unsigned int nSeed(seedHits.size());
  unsigned int nSeed(rseedHits.second-rseedHits.first);
  //seedHits.reserve(nSeed);
  TempTrajectoryContainer rebuiltTrajectories;

  for ( TempTrajectoryContainer::iterator it=result.begin();
	it!=result.end(); it++ ) {
    //
    // skip candidates which are not exceeding the seed size 
    // (e.g. because no Tracker layers outside seeding region) 
    //

    if ( it->measurements().size()<=startingTraj.measurements().size() ) {
      rebuiltTrajectories.push_back(std::move(*it));
      LogDebug("CkfPattern")<< "RebuildSeedingRegion skipped as in-out trajectory does not exceed seed size.";
      continue;
    }
    //
    // Refit - keep existing trajectory in case fit is not possible
    // or fails
    //

    auto && reFitted = backwardFit(*it,nSeed,fitter,seedHits);
    if unlikely( !reFitted.isValid() ) {
	rebuiltTrajectories.push_back(std::move(*it));
	LogDebug("CkfPattern")<< "RebuildSeedingRegion skipped as backward fit failed";
	//			    << "after reFitted.size() " << reFitted.size();
	continue;
      }
    //LogDebug("CkfPattern")<<"after reFitted.size() " << reFitted.size();
    //
    // Rebuild seeding part. In case it fails: keep initial trajectory
    // (better to drop it??)
    //
    int nRebuilt =
      rebuildSeedingRegion (seed, seedHits,reFitted,rebuiltTrajectories);

    if ( nRebuilt==0 && !theKeepOriginalIfRebuildFails ) it->invalidate();  // won't use original in-out track

    if ( nRebuilt<0 ) rebuiltTrajectories.push_back(std::move(*it));

  }
  //
  // Replace input trajectories with new ones
  //
  result.swap(rebuiltTrajectories);
  result.erase(std::remove_if( result.begin(),result.end(),
			       std::not1(std::mem_fun_ref(&TempTrajectory::isValid))),
	       result.end());
}

int
GroupedCkfTrajectoryBuilder::rebuildSeedingRegion(const TrajectorySeed&seed,
						  const std::vector<const TrackingRecHit*>& seedHits, 
						  TempTrajectory& candidate,
						  TempTrajectoryContainer& result) const 
{
  //
  // Starting from track found by in-out tracking phase, extrapolate it inwards through
  // the seeding region if possible in towards smaller Tracker radii, searching for additional
  // hits.
  // The resulting trajectories are returned in result,
  // the count is the return value.
  //
  TempTrajectoryContainer rebuiltTrajectories;
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
  const bool inOut = false;
  groupedLimitedCandidates(seed, candidate, nullptr, backwardPropagator(seed), inOut, rebuiltTrajectories);

  LogDebug("CkfPattern")<<" After backward building: "<<PrintoutHelper::dumpCandidates(rebuiltTrajectories);

  //
  // Check & count resulting candidates
  //
  int nrOfTrajectories(0);
  bool orig_ok = false;
  //const RecHitEqualByChannels recHitEqual(false,false);
  //vector<TM> oldMeasurements(candidate.measurements());
  for ( TempTrajectoryContainer::iterator it=rebuiltTrajectories.begin();
	it!=rebuiltTrajectories.end(); it++ ) {
    
    TempTrajectory::DataContainer newMeasurements(it->measurements());
    //
    // Verify presence of seeding hits?
    //
    if ( theRequireSeedHitsInRebuild ) {
      orig_ok = true;
      // no hits found (and possibly some invalid hits discarded): drop track
      if ( newMeasurements.size()<=candidate.measurements().size() ){  
	LogDebug("CkfPattern") << "newMeasurements.size()<=candidate.measurements().size()";
	continue;
      }	
      // verify presence of hits
      //GIO//if ( !verifyHits(newMeasurements.begin()+candidate.measurements().size(),
      //GIO//		       newMeasurements.end(),seedHits) ){
      if ( !verifyHits(newMeasurements.rbegin(), 
                       newMeasurements.size() - candidate.measurements().size(),
		       seedHits) ){
	LogDebug("CkfPattern")<< "seed hits not found in rebuild";
	continue; 
      }
    }
    //
    // construct final trajectory in the right order
    //
    // save & count result
    nrOfTrajectories++;
    result.emplace_back(seed.direction());
    TempTrajectory & reversedTrajectory = result.back();
    reversedTrajectory.setNLoops(it->nLoops());
    for (TempTrajectory::DataContainer::const_iterator im=newMeasurements.rbegin(), ed = newMeasurements.rend();
	 im != ed; --im ) {
      reversedTrajectory.push(*im);
    }
    
    LogDebug("CkgPattern")<<"New traj direction = " << reversedTrajectory.direction()<<"\n"
			  <<PrintoutHelper::dumpMeasurements(reversedTrajectory.measurements());
  } // rebuiltTrajectories

  // If nrOfTrajectories = 0 and orig_ok = true, this means that a track was actually found on the
  // out-in step (meeting those requirements) but did not have the seed hits in it.
  // In this case when we return we will go ahead and use the original in-out track.

  // If nrOfTrajectories = 0 and orig_ok = false, this means that the out-in step failed to
  // find any track.  Two cases are a technical failure in fitting the original seed hits or
  // because the track did not meet the out-in criteria (which may be stronger than the out-in
  // criteria).  In this case we will NOT allow the original in-out track to be used.

  if ( (nrOfTrajectories == 0) && orig_ok ) {
    nrOfTrajectories = -1;
  }
  return nrOfTrajectories;
}

TempTrajectory
GroupedCkfTrajectoryBuilder::backwardFit (TempTrajectory& candidate, unsigned int nSeed,
					  const TrajectoryFitter& fitter,
					  std::vector<const TrackingRecHit*>& remainingHits) const
{
  //
  // clear array of non-fitted hits
  //
  remainingHits.clear();
  //
  // skip candidates which are not exceeding the seed size
  // (e.g. Because no Tracker layers exist outside seeding region)
  //
  if unlikely( candidate.measurements().size()<=nSeed ) return TempTrajectory();

  LogDebug("CkfPattern")<<"nSeed " << nSeed << endl
			<< "Old traj direction = " << candidate.direction() << endl
			<<PrintoutHelper::dumpMeasurements(candidate.measurements());

  //
  // backward fit trajectory.
  // (Will try to fit only hits outside the seeding region. However,
  // if there are not enough of these, it will also use the seeding hits).
  //
  //   const unsigned int nHitAllMin(5);
  //   const unsigned int nHit2dMin(2);
  unsigned int nHit(0);    // number of valid hits after seeding region
  //unsigned int nHit2d(0);  // number of valid hits after seeding region with 2D info
  // use all hits except the first n (from seed), but require minimum
  // specified in configuration.
  //  Swapped over next two lines.
  unsigned int nHitMin = std::max(candidate.foundHits()-nSeed,theMinNrOfHitsForRebuild);
  //  unsigned int nHitMin = oldMeasurements.size()-nSeed;
  // we want to rebuild only if the number of VALID measurements excluding the seed measurements is higher than the cut
  if unlikely(nHitMin<theMinNrOfHitsForRebuild) return TempTrajectory();

  LogDebug("CkfPattern")/* << "nHitMin " << nHitMin*/ <<"Sizes: " << candidate.measurements().size() << " / ";
  //
  // create input trajectory for backward fit
  //
  Trajectory fwdTraj(oppositeDirection(candidate.direction()));
  fwdTraj.setNLoops(candidate.nLoops());
  //const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(), TrajectorySeed::recHitContainer(), oppositeDirection(candidate.direction()));
  //Trajectory fwdTraj(seed, oppositeDirection(candidate.direction()));

  const DetLayer* bwdDetLayer[candidate.measurements().size()];
  int nl=0;
  for ( auto const & tm : candidate.measurements() ) {
    const TrackingRecHit* hit = tm.recHitR().hit();
    //
    // add hits until required number is reached
    //
    if ( nHit<nHitMin ){//|| nHit2d<theMinNrOf2dHitsForRebuild ) {
      fwdTraj.push(tm);
      bwdDetLayer[nl++]=tm.layer();
      //
      // count valid / 2D hits
      //
      if likely( hit->isValid() ) {
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
  if unlikely( nHit<nHitMin ) return TempTrajectory();

  //
  // Do the backward fit (important: start from scaled, not random cov. matrix!)
  //
  TrajectoryStateOnSurface firstTsos(fwdTraj.firstMeasurement().updatedState());
  //cout << "firstTsos "<< firstTsos << endl;
  firstTsos.rescaleError(10.);
  //TrajectoryContainer bwdFitted(fitter.fit(fwdTraj.seed(),fwdTraj.recHits(),firstTsos));
  Trajectory && bwdFitted = fitter.fitOne(TrajectorySeed(PTrajectoryStateOnDet(), TrajectorySeed::recHitContainer(), oppositeDirection(candidate.direction())),
					  fwdTraj.recHits(),firstTsos);
  if unlikely(!bwdFitted.isValid()) return TempTrajectory();


  LogDebug("CkfPattern")<<"Obtained bwdFitted trajectory with measurement size " << bwdFitted.measurements().size();
  TempTrajectory fitted(fwdTraj.direction());
  fitted.setNLoops(fwdTraj.nLoops());
  vector<TM> const & tmsbf = bwdFitted.measurements();
  int iDetLayer=0;
  //this is ugly but the TM in the fitted track do not contain the DetLayer.
  //So we have to cache the detLayer pointers and replug them in.
  //For the backward building it would be enaugh to cache the last DetLayer, 
  //but for the intermediary cleaning we need all
  for ( vector<TM>::const_iterator im=tmsbf.begin();im!=tmsbf.end(); im++ ) {
    fitted.emplace( (*im).forwardPredictedState(),
		    (*im).backwardPredictedState(),
		    (*im).updatedState(),
		    (*im).recHit(),
		    (*im).estimate(),
		    bwdDetLayer[iDetLayer]
		    );
    
    LogDebug("CkfPattern")<<PrintoutHelper::dumpMeasurement(*im);
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
    lastBwdDetLayer));
  */
  
  
  return fitted;
}

bool
GroupedCkfTrajectoryBuilder::verifyHits (TempTrajectory::DataContainer::const_iterator rbegin,
                                         size_t maxDepth,
					 const std::vector<const TrackingRecHit*>& hits) const
{
  //
  // verify presence of the seeding hits
  //
  LogDebug("CkfPattern")<<"Checking for " << hits.size() << " hits in "
			<< maxDepth << " measurements" << endl;

  auto rend = rbegin; 
  while (maxDepth > 0) { --maxDepth; --rend; }
  for ( auto  ir=hits.begin();	ir!=hits.end(); ir++ ) {
    // assume that all seeding hits are valid!
    bool foundHit(false);
    for ( auto im=rbegin; im!=rend; --im ) {
      if ( im->recHit()->isValid() && (*ir)->sharesInput(im->recHit()->hit(), TrackingRecHit::some) ) {
	foundHit = true;
	break;
      }
    }
    if ( !foundHit )  return false;
  }
  return true;
}




