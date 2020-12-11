#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::string PrintoutHelper::dumpMeasurements(const std::vector<TrajectoryMeasurement>& v) {
  std::stringstream buffer;
  buffer << v.size() << " total measurements\n";
  std::vector<TrajectoryMeasurement>::const_iterator it = v.begin();
  for (; it != v.end(); ++it) {
    buffer << dumpMeasurement(*it);
    buffer << "\n";
  }
  return buffer.str();
}
std::string PrintoutHelper::dumpMeasurements(const cmsutils::bqueue<TrajectoryMeasurement>& v) {
  std::stringstream buffer;
  buffer << v.size() << " total measurements\n";
  cmsutils::bqueue<TrajectoryMeasurement>::const_iterator it = v.rbegin();
  for (; it != v.rend(); --it) {
    buffer << dumpMeasurement(*it);
    buffer << "\n";
  }
  return buffer.str();
}
std::string PrintoutHelper::dumpMeasurement(const TrajectoryMeasurement& tm) {
  std::stringstream buffer;
  buffer << "layer pointer: " << tm.layer() << "\n"
         << "estimate: " << tm.estimate() << "\n";
  if (tm.updatedState().isValid())
    buffer << "updated state: \n"
           << "x: " << tm.updatedState().globalPosition() << "\n"
           << "p: " << tm.updatedState().globalMomentum() << "\n";
  else if (tm.forwardPredictedState().isValid())
    buffer << "forward predicted state: \n"
           << "x: " << tm.forwardPredictedState().globalPosition() << "\n"
           << "p: " << tm.forwardPredictedState().globalMomentum() << "\n";
  else if (tm.predictedState().isValid())
    buffer << "predicted state: \n"
           << "x: " << tm.predictedState().globalPosition() << "\n"
           << "p: " << tm.predictedState().globalMomentum() << "\n";
  else
    buffer << "no valid state\n";
  buffer << "detId: " << tm.recHit()->geographicalId().rawId();
  if (tm.recHit()->isValid()) {
    buffer << "\n hit global x: " << tm.recHit()->globalPosition()
           << "\n hit global error: " << tm.recHit()->globalPositionError().matrix()
           << "\n hit local x:" << tm.recHit()->localPosition() << "\n hit local error"
           << tm.recHit()->localPositionError();
  } else
    buffer << "\n (-,-,-)";
  buffer << "\n fwdPred " << tm.forwardPredictedState().isValid() << "\n bwdPred "
         << tm.backwardPredictedState().isValid() << "\n upPred " << tm.updatedState().isValid();
  //SimIdPrinter()(tm.recHit());
  return buffer.str();
}

std::string PrintoutHelper::regressionTest(const TrackerGeometry& tracker, std::vector<Trajectory>& unsmoothedResult) {
  std::stringstream buffer;

  buffer << "number of finalTrajectories: " << unsmoothedResult.size() << std::endl;
  for (std::vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); it++) {
    if (it->lastMeasurement().updatedState().isValid()) {
      buffer << "candidate's n valid and invalid hit, chi2, pt, eta : " << it->foundHits() << " , " << it->lostHits()
             << " , " << it->chiSquared() << " , " << it->lastMeasurement().updatedState().globalMomentum().perp()
             << " , " << it->lastMeasurement().updatedState().globalMomentum().eta() << std::endl;
    } else if (it->lastMeasurement().predictedState().isValid()) {
      buffer << "candidate's n valid and invalid hit, chi2, pt, eta : " << it->foundHits() << " , " << it->lostHits()
             << " , " << it->chiSquared() << " , " << it->lastMeasurement().predictedState().globalMomentum().perp()
             << " , " << it->lastMeasurement().predictedState().globalMomentum().eta() << std::endl;
    } else
      buffer << "candidate with invalid last measurement state!" << std::endl;
  }
  buffer << "=================================================";
  buffer << "=========== Traj in details =====================\n";
  for (const auto& it : unsmoothedResult) {
    for (const auto& hit : it.measurements()) {
      buffer << "measurement : " << hit.recHit()->geographicalId().rawId() << std::endl;
    }
    buffer << "================\n";
  }
  return buffer.str();
}
