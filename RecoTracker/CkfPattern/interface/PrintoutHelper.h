#ifndef RecoTracker_CkfPattern_PrintoutHelper_h
#define RecoTracker_CkfPattern_PrintoutHelper_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/bqueue.h"

class TrackerGeometry;

class PrintoutHelper {
public:
  template <class collection>
  static std::string dumpCandidates(collection& candidates);
  template <class Candidate>
  static std::string dumpCandidate(const Candidate& candidate, bool showErrors = false);
  static std::string dumpMeasurements(const std::vector<TrajectoryMeasurement>& v);
  static std::string dumpMeasurements(const cmsutils::bqueue<TrajectoryMeasurement>& v);
  static std::string dumpMeasurement(const TrajectoryMeasurement& tm);
  static std::string regressionTest(const TrackerGeometry& tracker, std::vector<Trajectory>& unsmoothedResult);
};

template <class Candidate>
std::string PrintoutHelper::dumpCandidate(const Candidate& traj, bool showErrors) {
  // does not work....
  LogDebug("PrintoutHelperError") << "switching on error printout" << (showErrors = true);

  std::stringstream buffer;
  if (!traj.measurements().empty()) {
    const TrajectoryMeasurement& last = traj.lastMeasurement();

    buffer << "with: " << traj.measurements().size() << " measurements." << traj.lostHits() << " lost, "
           << traj.foundHits() << " found, " << traj.trailingFoundHits() << " trailing, " << traj.cccBadHits()
           << " badCC, "
           << "chi2=" << traj.chiSquared() << ' ' << int(traj.nLoops()) << " loops\n";
    if (last.updatedState().isValid()) {
      const TrajectoryStateOnSurface& tsos = last.updatedState();
      if (showErrors)
        buffer << "Last [Updated] state\n : " << tsos << "\n";
      else
        buffer << "Last [Updated] state\n x: " << tsos.globalPosition() << "\n p: " << tsos.globalMomentum() << "\n";
    } else if (last.forwardPredictedState().isValid()) {
      const TrajectoryStateOnSurface& tsos = last.forwardPredictedState();
      if (showErrors)
        buffer << "Last [fwdPredicted] state\n : " << tsos << "\n";
      else
        buffer << "Last [fwdPredicted] state\n x: " << tsos.globalPosition() << "\n p: " << tsos.globalMomentum()
               << "\n";
    } else if (last.predictedState().isValid()) {
      const TrajectoryStateOnSurface& tsos = last.predictedState();
      if (showErrors)
        buffer << "Last [Predicted] state\n : " << tsos << "\n";
      else
        buffer << "Last [Predicted] state\n x: " << tsos.globalPosition() << "\n p: " << tsos.globalMomentum() << "\n";
    }
    buffer << " hit is: " << (last.recHit()->isValid() ? "valid" : "invalid") << "\n";
    if (last.recHit()->isValid()) {
      buffer << "on detId: " << last.recHit()->geographicalId().rawId() << "\n";
      buffer << "gp: " << last.recHit()->globalPosition() << "\n";
    }
  } else {
    buffer << " no measurement. \n";
  }
  return buffer.str();
}

template <class collection>
std::string PrintoutHelper::dumpCandidates(collection& candidates) {
  std::stringstream buffer;
  buffer << "\n____________________________\n";
  unsigned int ic = 0;
  for (auto const& traj : candidates) {
    buffer << ic++ << "] " << (traj.isValid() ? "valid " : "invalid ");
    buffer << PrintoutHelper::dumpCandidate(traj);
  }
  buffer << "\n____________________________\n";
  return buffer.str();
}

#endif
