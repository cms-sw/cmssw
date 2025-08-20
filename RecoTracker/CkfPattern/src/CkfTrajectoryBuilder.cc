#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

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

#include "RecoTracker/CkfPattern/interface/TrajCandLess.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

using namespace std;

CkfTrajectoryBuilder::CkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : CkfTrajectoryBuilder(conf,
                           iC,
                           BaseCkfTrajectoryBuilder::createTrajectoryFilter(
                               conf.getParameter<edm::ParameterSet>("trajectoryFilter"), iC)) {}

CkfTrajectoryBuilder::CkfTrajectoryBuilder(const edm::ParameterSet& conf,
                                           edm::ConsumesCollector iC,
                                           std::unique_ptr<TrajectoryFilter> filter)
    : BaseCkfTrajectoryBuilder(conf, iC, std::move(filter)) {
  theMaxCand = conf.getParameter<int>("maxCand");
  theLostHitPenalty = conf.getParameter<double>("lostHitPenalty");
  theFoundHitBonus = conf.getParameter<double>("foundHitBonus");
  theMinHitForDoubleBonus = conf.getParameter<int>("minHitForDoubleBonus");
  theIntermediateCleaning = conf.getParameter<bool>("intermediateCleaning");
  theAlwaysUseInvalidHits = conf.getParameter<bool>("alwaysUseInvalidHits");
}

void CkfTrajectoryBuilder::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  BaseCkfTrajectoryBuilder::fillPSetDescription(iDesc);
  iDesc.add<int>("maxCand", 5);
  iDesc.add<double>("lostHitPenalty", 30.);
  iDesc.add<double>("foundHitBonus", 0.);
  iDesc.add<int>("minHitForDoubleBonus", 8888);
  iDesc.add<bool>("intermediateCleaning", true);
  iDesc.add<bool>("alwaysUseInvalidHits", true);

  edm::ParameterSetDescription psdTF;
  psdTF.addNode(edm::PluginDescription<TrajectoryFilterFactory>("ComponentType", true));
  iDesc.add<edm::ParameterSetDescription>("trajectoryFilter", psdTF);
}

void CkfTrajectoryBuilder::setEvent_(const edm::Event& event, const edm::EventSetup& iSetup) {}

CkfTrajectoryBuilder::TrajectoryContainer CkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed) const {
  TrajectoryContainer result;
  result.reserve(5);
  trajectories(seed, result);
  return result;
}

void CkfTrajectoryBuilder::trajectories(const TrajectorySeed& seed,
                                        CkfTrajectoryBuilder::TrajectoryContainer& result) const {
  unsigned int tmp;
  buildTrajectories(seed, result, tmp, nullptr);
}

void CkfTrajectoryBuilder::buildTrajectories(const TrajectorySeed& seed,
                                             TrajectoryContainer& result,
                                             unsigned int& nCandPerSeed,
                                             const TrajectoryFilter*) const {
  if (theMeasurementTracker == nullptr) {
    throw cms::Exception("LogicError")
        << "Asking to create trajectories to an un-initialized CkfTrajectoryBuilder.\nYou have to call clone(const "
           "MeasurementTrackerEvent *data) and then call trajectories on it instead.\n";
  }

  TempTrajectory startingTraj = createStartingTrajectory(seed);

  nCandPerSeed = limitedCandidates(seed, startingTraj, result);
}

unsigned int CkfTrajectoryBuilder::limitedCandidates(const TrajectorySeed& seed,
                                                     TempTrajectory& startingTraj,
                                                     TrajectoryContainer& result) const {
  TempTrajectoryContainer candidates;
  candidates.push_back(startingTraj);
  std::shared_ptr<const TrajectorySeed> sharedSeed(new TrajectorySeed(seed));
  return limitedCandidates(sharedSeed, candidates, result);
}

unsigned int CkfTrajectoryBuilder::limitedCandidates(const std::shared_ptr<const TrajectorySeed>& sharedSeed,
                                                     TempTrajectoryContainer& candidates,
                                                     TrajectoryContainer& result) const {
  unsigned int nIter = 1;
  unsigned int nCands = 0;  // ignore startingTraj
  unsigned int prevNewCandSize = 0;
  TempTrajectoryContainer newCand;  // = TrajectoryContainer();
  newCand.reserve(theMaxCand);

  auto score = [&](TempTrajectory const& a) {
    auto bonus = theFoundHitBonus;
    bonus += a.foundHits() > theMinHitForDoubleBonus ? bonus : 0;
    return a.chiSquared() + a.lostHits() * theLostHitPenalty - bonus * a.foundHits();
  };

  auto trajCandLess = [&](TempTrajectory const& a, TempTrajectory const& b) { return score(a) < score(b); };

  while (!candidates.empty()) {
    newCand.clear();
    bool full = false;
    for (auto traj = candidates.begin(); traj != candidates.end(); traj++) {
      std::vector<TM> meas;
      findCompatibleMeasurements(*sharedSeed, *traj, meas);

      // --- method for debugging
      if (!analyzeMeasurementsDebugger(
              *traj, meas, theMeasurementTracker, forwardPropagator(*sharedSeed), theEstimator, theTTRHBuilder))
        return nCands;
      // ---

      if (meas.empty()) {
        addToResult(sharedSeed, *traj, result);
      } else {
        std::vector<TM>::const_iterator last;
        if (theAlwaysUseInvalidHits)
          last = meas.end();
        else {
          if (meas.front().recHit()->isValid()) {
            last = find_if(meas.begin(), meas.end(), [](auto const& meas) { return !meas.recHit()->isValid(); });
          } else
            last = meas.end();
        }

        for (auto itm = meas.begin(); itm != last; itm++) {
          TempTrajectory newTraj = *traj;
          updateTrajectory(newTraj, std::move(*itm));

          if (toBeContinued(newTraj)) {
            if (full) {
              bool better = trajCandLess(newTraj, newCand.front());
              if (better) {
                // replace worst
                std::pop_heap(newCand.begin(), newCand.end(), trajCandLess);
                newCand.back().swap(newTraj);
                std::push_heap(newCand.begin(), newCand.end(), trajCandLess);
              }  // else? no need to add it just to remove it later!
            } else {
              newCand.push_back(std::move(newTraj));
              full = (int)newCand.size() == theMaxCand;
              if (full)
                std::make_heap(newCand.begin(), newCand.end(), trajCandLess);
            }
          } else {
            addToResult(sharedSeed, newTraj, result);
            //// don't know yet
          }
        }
      }

      // account only new candidates, i.e.
      // - 1 candidate -> 1 candidate, don't increase count
      // - 1 candidate -> 2 candidates, increase count by 1
      nCands += newCand.size() - prevNewCandSize;
      prevNewCandSize = newCand.size();

      assert((int)newCand.size() <= theMaxCand);
      if (full)
        assert((int)newCand.size() == theMaxCand);
    }  // end loop on candidates

    // no reason to sort  (no sorting in Grouped version!)
    if (theIntermediateCleaning)
      IntermediateTrajectoryCleaner::clean(newCand);

    candidates.swap(newCand);

    LogDebug("CkfPattern") << result.size() << " candidates after " << nIter++ << " CKF iteration: \n"
                           << PrintoutHelper::dumpCandidates(result) << "\n " << candidates.size()
                           << " running candidates are: \n"
                           << PrintoutHelper::dumpCandidates(candidates);
  }
  return nCands;
}

void CkfTrajectoryBuilder::updateTrajectory(TempTrajectory& traj, TM&& tm) const {
  auto&& predictedState = tm.predictedState();
  auto&& hit = tm.recHit();
  if (hit->isValid()) {
    auto&& upState = theUpdator->update(predictedState, *hit);
    traj.emplace(predictedState, std::move(upState), hit, tm.estimate(), tm.layer());
  } else {
    traj.emplace(predictedState, hit, 0, tm.layer());
  }
}

void CkfTrajectoryBuilder::findCompatibleMeasurements(const TrajectorySeed& seed,
                                                      const TempTrajectory& traj,
                                                      std::vector<TrajectoryMeasurement>& result) const {
  int invalidHits = 0;
  //Use findStateAndLayers which handles the hitless seed use case
  std::pair<TSOS, std::vector<const DetLayer*> >&& stateAndLayers = findStateAndLayers(seed, traj);
  if (stateAndLayers.second.empty())
    return;

  auto layerBegin = stateAndLayers.second.begin();
  auto layerEnd = stateAndLayers.second.end();
  LogDebug("CkfPattern") << "looping on " << stateAndLayers.second.size() << " layers.";
  const Propagator* fwdPropagator = forwardPropagator(seed);
  for (auto il = layerBegin; il != layerEnd; il++) {
    LogDebug("CkfPattern") << "looping on a layer in findCompatibleMeasurements.\n last layer: " << traj.lastLayer()
                           << " current layer: " << (*il);

    TSOS stateToUse = stateAndLayers.first;
    //Added protection before asking for the lastLayer on the trajectory
    if UNLIKELY (!traj.empty() && (*il) == traj.lastLayer()) {
      LogDebug("CkfPattern") << " self propagating in findCompatibleMeasurements.\n from: \n" << stateToUse;
      //self navigation case
      // go to a middle point first
      TransverseImpactPointExtrapolator middle;
      GlobalPoint center(0, 0, 0);
      stateToUse = middle.extrapolate(stateToUse, center, *fwdPropagator);

      if (!stateToUse.isValid())
        continue;
      LogDebug("CkfPattern") << "to: " << stateToUse;
    }

    LayerMeasurements layerMeasurements(theMeasurementTracker->measurementTracker(), *theMeasurementTracker);
    std::vector<TrajectoryMeasurement>&& tmp =
        layerMeasurements.measurements((**il), stateToUse, *fwdPropagator, *theEstimator);

    if (!tmp.empty()) {
      if (result.empty())
        result.swap(tmp);
      else {
        // keep one dummy TM at the end, skip the others
        result.insert(
            result.end() - invalidHits, std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
      }
      invalidHits++;
    }
  }

  // sort the final result, keep dummy measurements at the end
  if (result.size() > 1) {
    std::sort(result.begin(), result.end() - invalidHits, TrajMeasLessEstim());
  }

  LogDebug("CkfPattern") << "starting from:\n"
                         << "x: " << stateAndLayers.first.globalPosition() << "\n"
                         << "p: " << stateAndLayers.first.globalMomentum() << "\n"
                         << PrintoutHelper::dumpMeasurements(result);

#ifdef DEBUG_INVALID
  bool afterInvalid = false;
  for (vector<TM>::const_iterator i = result.begin(); i != result.end(); i++) {
    if (!i->recHit().isValid())
      afterInvalid = true;
    if (afterInvalid && i->recHit().isValid()) {
      edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: valid hit after invalid!";
    }
  }
#endif
}
