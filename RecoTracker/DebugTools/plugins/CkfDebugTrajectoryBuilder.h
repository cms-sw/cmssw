#ifndef CkfDebugTrajectoryBuilder_H
#define CkfDebugTrajectoryBuilder_H

#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "CkfDebugger.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class CkfDebugTrajectoryBuilder : public CkfTrajectoryBuilder {
public:
  CkfDebugTrajectoryBuilder(const edm::ParameterSet& conf)
      : CkfTrajectoryBuilder(conf, std::unique_ptr<TrajectoryFilter>{}) {
    //edm::LogVerbatim("CkfDebugger") <<"CkfDebugTrajectoryBuilder::CkfDebugTrajectoryBuilder";
  }

  void setDebugger(CkfDebugger* dbg) const override { theDbg = dbg; }
  virtual CkfDebugger* debugger() const { return theDbg; }

private:
  mutable CkfDebugger* theDbg;
  bool analyzeMeasurementsDebugger(TempTrajectory& traj,
                                   const std::vector<TM>& meas,
                                   const MeasurementTrackerEvent* theMeasurementTracker,
                                   const Propagator* theForwardPropagator,
                                   const Chi2MeasurementEstimatorBase* theEstimator,
                                   const TransientTrackingRecHitBuilder* theTTRHBuilder) const override {
    return theDbg->analyseCompatibleMeasurements(
        traj.toTrajectory(), meas, theMeasurementTracker, theForwardPropagator, theEstimator, theTTRHBuilder);
  };
  bool analyzeMeasurementsDebugger(Trajectory& traj,
                                   const std::vector<TM>& meas,
                                   const MeasurementTrackerEvent* theMeasurementTracker,
                                   const Propagator* theForwardPropagator,
                                   const Chi2MeasurementEstimatorBase* theEstimator,
                                   const TransientTrackingRecHitBuilder* theTTRHBuilder) const override {
    return theDbg->analyseCompatibleMeasurements(
        traj, meas, theMeasurementTracker, theForwardPropagator, theEstimator, theTTRHBuilder);
  };
  void fillSeedHistoDebugger(std::vector<TrajectoryMeasurement>::iterator begin,
                             std::vector<TrajectoryMeasurement>::iterator end) const override {
    //edm::LogVerbatim("CkfDebugger") <<"CkfDebugTrajectoryBuilder::fillSeedHistoDebugger "<<theDbg;
    if (end - begin >= 2)
      theDbg->fillSeedHist(begin->recHit(), (begin + 1)->recHit(), (begin + 1)->updatedState());
  };
};
#endif
