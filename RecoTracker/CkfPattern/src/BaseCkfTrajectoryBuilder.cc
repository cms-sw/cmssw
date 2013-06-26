#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

  
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


BaseCkfTrajectoryBuilder::
BaseCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
			 const TrajectoryStateUpdator*         updator,
			 const Propagator*                     propagatorAlong,
			 const Propagator*                     propagatorOpposite,
			 const Chi2MeasurementEstimatorBase*   estimator,
			 const TransientTrackingRecHitBuilder* recHitBuilder,
			 const MeasurementTracker*             measurementTracker,
			 const TrajectoryFilter*               filter,
                         const TrajectoryFilter*               inOutFilter):
  theUpdator(updator),
  thePropagatorAlong(propagatorAlong),thePropagatorOpposite(propagatorOpposite),
  theEstimator(estimator),theTTRHBuilder(recHitBuilder),
  theMeasurementTracker(measurementTracker),
  theLayerMeasurements(new LayerMeasurements(theMeasurementTracker)),
  theForwardPropagator(0),theBackwardPropagator(0),
  theFilter(filter),
  theInOutFilter(inOutFilter)
{
  if (conf.exists("clustersToSkip")){
    skipClusters_=true;
    clustersToSkip_=conf.getParameter<edm::InputTag>("clustersToSkip");
  }
  else
    skipClusters_=false;
}
 
BaseCkfTrajectoryBuilder::~BaseCkfTrajectoryBuilder(){
  delete theLayerMeasurements;
}


void
BaseCkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed,  TempTrajectory & result) const
{
  

  TrajectorySeed::range hitRange = seed.recHits();

  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet(pState.detId());
  TSOS outerState = trajectoryStateTransform::transientState(pState, &(gdet->surface()),
							     theForwardPropagator->magneticField());


  for (TrajectorySeed::const_iterator ihit = hitRange.first; ihit != hitRange.second; ihit++) {
 
   TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = recHit->det();
 
    const DetLayer* hitLayer = 
      theMeasurementTracker->geometricSearchTracker()->detLayer(ihit->geographicalId());

    TSOS invalidState( hitGeomDet->surface());
    if (ihit == hitRange.second - 1) {
      // the seed trajectory state should correspond to this hit
      if (&gdet->surface() != &hitGeomDet->surface()) {
	edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
	return; // FIXME: should throw exception
      }

      //TSOS updatedState = outerstate;
      result.emplace(invalidState, outerState, recHit, 0, hitLayer);
    }
    else {
      TSOS innerState   = theBackwardPropagator->propagate(outerState,hitGeomDet->surface());
      if(innerState.isValid()) {
	TSOS innerUpdated = theUpdator->update(innerState,*recHit);
	result.emplace(invalidState, innerUpdated, recHit, 0, hitLayer);
      }
    }
  }

  // method for debugging
  // fix somehow
  // fillSeedHistoDebugger(result.begin(),result.end());

}


TempTrajectory BaseCkfTrajectoryBuilder::
createStartingTrajectory( const TrajectorySeed& seed) const
{
  TempTrajectory result(seed.direction());
  if (  seed.direction() == alongMomentum) {
    theForwardPropagator = &(*thePropagatorAlong);
    theBackwardPropagator = &(*thePropagatorOpposite);
  }
  else {
    theForwardPropagator = &(*thePropagatorOpposite);
    theBackwardPropagator = &(*thePropagatorAlong);
  }

  seedMeasurements(seed, result);

  LogDebug("CkfPattern")
    <<" initial trajectory from the seed: "<<PrintoutHelper::dumpCandidate(result,true);
  
  return result;
}


bool BaseCkfTrajectoryBuilder::toBeContinued (TempTrajectory& traj, bool inOut) const
{
  if (traj.measurements().size() > 400) {
    edm::LogError("BaseCkfTrajectoryBuilder_InfiniteLoop");
    LogTrace("BaseCkfTrajectoryBuilder_InfiniteLoop") << 
              "Cropping Track After 400 Measurements:\n" <<
              "   Last predicted state: " << traj.lastMeasurement().predictedState() << "\n" <<
              "   Last layer subdetector: " << (traj.lastLayer() ? traj.lastLayer()->subDetector() : -1) << "\n" <<
              "   Found hits: " << traj.foundHits() << ", lost hits: " << traj.lostHits() << "\n\n";
    return false;
  }
  // Called after each new hit is added to the trajectory, to see if it is 
  // worth continuing to build this track candidate.
  if (inOut) {
    if (theInOutFilter == 0) edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: trying to use dedicated filter for in-out tracking phase, when none specified";
    return theInOutFilter->toBeContinued(traj);
  } else {
    return theFilter->toBeContinued(traj);
  }
}


 bool BaseCkfTrajectoryBuilder::qualityFilter( const TempTrajectory& traj, bool inOut) const
{
  // Called after building a trajectory is completed, to see if it is good enough
  // to keep.
  if (inOut) {
    if (theInOutFilter == 0) edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: trying to use dedicated filter for in-out tracking phase, when none specified";
    return theInOutFilter->qualityFilter(traj);
  } else {
    return theFilter->qualityFilter(traj);
  }
}


void 
BaseCkfTrajectoryBuilder::addToResult (boost::shared_ptr<const TrajectorySeed> const & seed, TempTrajectory& tmptraj, 
				       TrajectoryContainer& result,
                                       bool inOut) const
{
  // quality check
  if ( !qualityFilter(tmptraj, inOut) )  return;
  Trajectory traj = tmptraj.toTrajectory();
  traj.setSharedSeed(seed);
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  LogDebug("CkfPattern")<<inOut<<"=inOut option. pushing a Trajectory with: "<<traj.foundHits()<<" found hits. "<<traj.lostHits()
			<<" lost hits. Popped :"<<(tmptraj.measurements().size())-(traj.measurements().size())<<" hits.";
  result.push_back(std::move(traj));
}


void 
BaseCkfTrajectoryBuilder::addToResult (TempTrajectory const & tmptraj, 
				       TempTrajectoryContainer& result,
                                       bool inOut) const
{
  // quality check
  if ( !qualityFilter(tmptraj, inOut) )  return;
  // discard latest dummy measurements
  TempTrajectory traj = tmptraj;
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid()) traj.pop();
  LogDebug("CkfPattern")<<inOut<<"=inOut option. pushing a TempTrajectory with: "<<traj.foundHits()<<" found hits. "<<traj.lostHits()
			<<" lost hits. Popped :"<<(tmptraj.measurements().size())-(traj.measurements().size())<<" hits.";
  result.push_back(std::move(traj));
}

void 
BaseCkfTrajectoryBuilder::moveToResult (TempTrajectory&& traj, 
				       TempTrajectoryContainer& result,
                                       bool inOut) const
{
  // quality check
  if ( !qualityFilter(traj, inOut) )  return;
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHitR().isValid()) traj.pop();
  LogDebug("CkfPattern")<<inOut<<"=inOut option. pushing a TempTrajectory with: "<<traj.foundHits()<<" found hits. "<<traj.lostHits();
    //			<<" lost hits. Popped :"<<(ttraj.measurements().size())-(traj.measurements().size())<<" hits.";
  result.push_back(std::move(traj));
}



BaseCkfTrajectoryBuilder::StateAndLayers
BaseCkfTrajectoryBuilder::findStateAndLayers(const TrajectorySeed& seed, const TempTrajectory& traj) const
{
  if (traj.empty())
    {
      //set the currentState to be the one from the trajectory seed starting point
      PTrajectoryStateOnDet const & ptod = seed.startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theMeasurementTracker->geomTracker()->idToDet(id);                    
      const Surface * surface=&g->surface();
      
      
      TSOS currentState(trajectoryStateTransform::transientState(ptod,surface,theForwardPropagator->magneticField()));      
      const DetLayer* lastLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(id);      
      return StateAndLayers(currentState,lastLayer->nextLayers( *currentState.freeState(), traj.direction()) );
    }
  else
    {  
      TSOS const & currentState = traj.lastMeasurement().updatedState();
      return StateAndLayers(currentState,traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction()) );
    }
}

BaseCkfTrajectoryBuilder::StateAndLayers
BaseCkfTrajectoryBuilder::findStateAndLayers(const TempTrajectory& traj) const{
  assert(!traj.empty());
 
  TSOS const & currentState = traj.lastMeasurement().updatedState();
  return StateAndLayers(currentState,traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction()) );
}



void BaseCkfTrajectoryBuilder::setEvent(const edm::Event& event) const
  {
    theMeasurementTracker->update(event);
    if (skipClusters_)
      theMeasurementTracker->setClusterToSkip(clustersToSkip_,event);
  }

void BaseCkfTrajectoryBuilder::unset() const
{
  if (skipClusters_)
    theMeasurementTracker->unsetClusterToSkip();
}
