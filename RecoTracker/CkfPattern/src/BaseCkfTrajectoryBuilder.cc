#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

BaseCkfTrajectoryBuilder::BaseCkfTrajectoryBuilder(const edm::ParameterSet& conf,
                                                   std::unique_ptr<TrajectoryFilter> filter,
                                                   std::unique_ptr<TrajectoryFilter> inOutFilter)
    : theSeedAs5DHit(conf.getParameter<bool>("seedAs5DHit")),
      theFilter(std::move(filter)),
      theInOutFilter(std::move(inOutFilter)),
      theUpdatorName(conf.getParameter<std::string>("updator")),
      thePropagatorAlongName(conf.getParameter<std::string>("propagatorAlong")),
      thePropagatorOppositeName(conf.getParameter<std::string>("propagatorOpposite")),
      theEstimatorName(conf.getParameter<std::string>("estimator")),
      theRecHitBuilderName(conf.getParameter<std::string>("TTRHBuilder")) {
  if (conf.exists("clustersToSkip"))
    edm::LogError("BaseCkfTrajectoryBuilder")
        << "ERROR: " << typeid(*this).name() << " has a clustersToSkip parameter set";
}

BaseCkfTrajectoryBuilder::~BaseCkfTrajectoryBuilder() {}

std::unique_ptr<TrajectoryFilter> BaseCkfTrajectoryBuilder::createTrajectoryFilter(const edm::ParameterSet& pset,
                                                                                   edm::ConsumesCollector& iC) {
  return TrajectoryFilterFactory::get()->create(pset.getParameter<std::string>("ComponentType"), pset, iC);
}

#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit5DParamConstraint.h"
void BaseCkfTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed, TempTrajectory& result, bool as5D) const {
  TrajectorySeed::range hitRange = seed.recHits();

  PTrajectoryStateOnDet pState(seed.startingState());
  const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet(pState.detId());
  TSOS outerState =
      trajectoryStateTransform::transientState(pState, &(gdet->surface()), forwardPropagator(seed)->magneticField());

  if (as5D) {
    TrackingRecHit::RecHitPointer recHit(new TRecHit5DParamConstraint(*gdet, outerState));
    TSOS invalidState(gdet->surface());
    auto hitLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(pState.detId());
    result.emplace(invalidState, outerState, recHit, 0, hitLayer);
    return;
  }

  for (TrajectorySeed::const_iterator ihit = hitRange.first; ihit != hitRange.second; ihit++) {
    TrackingRecHit::RecHitPointer recHit = ihit->cloneSH();
    const GeomDet* hitGeomDet = recHit->det();

    const DetLayer* hitLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(ihit->geographicalId());

    TSOS invalidState(hitGeomDet->surface());
    if (ihit == hitRange.second - 1) {
      // the seed trajectory state should correspond to this hit
      if (&gdet->surface() != &hitGeomDet->surface()) {
        edm::LogError("CkfPattern")
            << "CkfTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
        return;  // FIXME: should throw exception
      }

      //TSOS updatedState = outerstate;
      result.emplace(invalidState, outerState, recHit, 0, hitLayer);
    } else {
      TSOS innerState = backwardPropagator(seed)->propagate(outerState, hitGeomDet->surface());

      // try to recover if propagation failed
      if
        UNLIKELY(!innerState.isValid())
      innerState = trajectoryStateTransform::transientState(
          pState, &(hitGeomDet->surface()), forwardPropagator(seed)->magneticField());

      if (innerState.isValid()) {
        TSOS innerUpdated = theUpdator->update(innerState, *recHit);
        result.emplace(invalidState, innerUpdated, recHit, 0, hitLayer);
      }
    }
  }

  // method for debugging
  // fix somehow
  // fillSeedHistoDebugger(result.begin(),result.end());
}

TempTrajectory BaseCkfTrajectoryBuilder::createStartingTrajectory(const TrajectorySeed& seed) const {
  TempTrajectory result(seed.direction(), seed.nHits());
  seedMeasurements(seed, result, theSeedAs5DHit);

  LogDebug("CkfPattern") << " initial trajectory from the seed: " << PrintoutHelper::dumpCandidate(result, true);

  return result;
}

bool BaseCkfTrajectoryBuilder::toBeContinued(TempTrajectory& traj, bool inOut) const {
  if
    UNLIKELY(traj.measurements().size() > 400) {
      edm::LogError("BaseCkfTrajectoryBuilder_InfiniteLoop");
      LogTrace("BaseCkfTrajectoryBuilder_InfiniteLoop")
          << "Cropping Track After 400 Measurements:\n"
          << "   Last predicted state: " << traj.lastMeasurement().predictedState() << "\n"
          << "   Last layer subdetector: " << (traj.lastLayer() ? traj.lastLayer()->subDetector() : -1) << "\n"
          << "   Found hits: " << traj.foundHits() << ", lost hits: " << traj.lostHits() << "\n\n";
      return false;
    }
  // Called after each new hit is added to the trajectory, to see if it is
  // worth continuing to build this track candidate.
  if (inOut) {
    // if (theInOutFilter == 0) edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: trying to use dedicated filter for in-out tracking phase, when none specified";
    return theInOutFilter->toBeContinued(traj);
  } else {
    return theFilter->toBeContinued(traj);
  }
}

bool BaseCkfTrajectoryBuilder::qualityFilter(const TempTrajectory& traj, bool inOut) const {
  // Called after building a trajectory is completed, to see if it is good enough
  // to keep.
  if (inOut) {
    // if (theInOutFilter == 0) edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: trying to use dedicated filter for in-out tracking phase, when none specified";
    return theInOutFilter->qualityFilter(traj);
  } else {
    return theFilter->qualityFilter(traj);
  }
}

void BaseCkfTrajectoryBuilder::addToResult(std::shared_ptr<const TrajectorySeed> const& seed,
                                           TempTrajectory& tmptraj,
                                           TrajectoryContainer& result,
                                           bool inOut) const {
  // quality check
  if (!qualityFilter(tmptraj, inOut))
    return;
  Trajectory traj = tmptraj.toTrajectory();
  traj.setSharedSeed(seed);
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid())
    traj.pop();
  LogDebug("CkfPattern") << inOut << "=inOut option. pushing a Trajectory with: " << traj.foundHits() << " found hits. "
                         << traj.lostHits()
                         << " lost hits. Popped :" << (tmptraj.measurements().size()) - (traj.measurements().size())
                         << " hits.";
  result.push_back(std::move(traj));
}

void BaseCkfTrajectoryBuilder::addToResult(TempTrajectory const& tmptraj,
                                           TempTrajectoryContainer& result,
                                           bool inOut) const {
  // quality check
  if (!qualityFilter(tmptraj, inOut))
    return;
  // discard latest dummy measurements
  TempTrajectory traj = tmptraj;
  while (!traj.empty() && !traj.lastMeasurement().recHit()->isValid())
    traj.pop();
  LogDebug("CkfPattern") << inOut << "=inOut option. pushing a TempTrajectory with: " << traj.foundHits()
                         << " found hits. " << traj.lostHits()
                         << " lost hits. Popped :" << (tmptraj.measurements().size()) - (traj.measurements().size())
                         << " hits.";
  result.push_back(std::move(traj));
}

void BaseCkfTrajectoryBuilder::moveToResult(TempTrajectory&& traj, TempTrajectoryContainer& result, bool inOut) const {
  // quality check
  if (!qualityFilter(traj, inOut))
    return;
  // discard latest dummy measurements
  while (!traj.empty() && !traj.lastMeasurement().recHitR().isValid())
    traj.pop();
  LogDebug("CkfPattern") << inOut << "=inOut option. pushing a TempTrajectory with: " << traj.foundHits()
                         << " found hits. " << traj.lostHits();
  //			<<" lost hits. Popped :"<<(ttraj.measurements().size())-(traj.measurements().size())<<" hits.";
  result.push_back(std::move(traj));
}

BaseCkfTrajectoryBuilder::StateAndLayers BaseCkfTrajectoryBuilder::findStateAndLayers(
    const TrajectorySeed& seed, const TempTrajectory& traj) const {
  if (traj.empty()) {
    //set the currentState to be the one from the trajectory seed starting point
    PTrajectoryStateOnDet const& ptod = seed.startingState();
    DetId id(ptod.detId());
    const GeomDet* g = theMeasurementTracker->geomTracker()->idToDet(id);
    const Surface* surface = &g->surface();

    TSOS currentState(
        trajectoryStateTransform::transientState(ptod, surface, forwardPropagator(seed)->magneticField()));
    const DetLayer* lastLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(id);
    return StateAndLayers(currentState,
                          theNavigationSchool->nextLayers(*lastLayer, *currentState.freeState(), traj.direction()));
  } else {
    TSOS const& currentState = traj.lastMeasurement().updatedState();
    return StateAndLayers(
        currentState, theNavigationSchool->nextLayers(*traj.lastLayer(), *currentState.freeState(), traj.direction()));
  }
}

void BaseCkfTrajectoryBuilder::setData(const MeasurementTrackerEvent* data) {
  // possibly do some sanity check here
  theMeasurementTracker = data;
}

void BaseCkfTrajectoryBuilder::setEvent(const edm::Event& event) const {
  std::cerr << "ERROR SetEvent called on " << typeid(*this).name()
            << (theMeasurementTracker ? " with valid " : "witout any ") << "MeasurementTrackerEvent" << std::endl;
}

void BaseCkfTrajectoryBuilder::unset() const {
  std::cerr << "ERROR unSet called on " << typeid(*this).name()
            << (theMeasurementTracker ? " with valid " : "witout any ") << "MeasurementTrackerEvent" << std::endl;
}

void BaseCkfTrajectoryBuilder::setEvent(const edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        const MeasurementTrackerEvent* data) {
  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  edm::ESHandle<Propagator> propagatorAlongHandle;
  edm::ESHandle<Propagator> propagatorOppositeHandle;
  edm::ESHandle<Chi2MeasurementEstimatorBase> estimatorHandle;
  edm::ESHandle<TransientTrackingRecHitBuilder> recHitBuilderHandle;

  iSetup.get<TrackingComponentsRecord>().get(theUpdatorName, updatorHandle);
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorAlongName, propagatorAlongHandle);
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorOppositeName, propagatorOppositeHandle);
  iSetup.get<TrackingComponentsRecord>().get(theEstimatorName, estimatorHandle);
  iSetup.get<TransientRecHitRecord>().get(theRecHitBuilderName, recHitBuilderHandle);

  theUpdator = updatorHandle.product();
  thePropagatorAlong = propagatorAlongHandle.product();
  thePropagatorOpposite = propagatorOppositeHandle.product();
  theEstimator = estimatorHandle.product();
  theTTRHBuilder = recHitBuilderHandle.product();

  setData(data);
  if (theFilter)
    theFilter->setEvent(iEvent, iSetup);
  if (theInOutFilter)
    theInOutFilter->setEvent(iEvent, iSetup);
  setEvent_(iEvent, iSetup);
}
