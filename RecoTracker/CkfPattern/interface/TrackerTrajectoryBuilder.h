#ifndef RecoTracker_CkfPattern_TrackerTrajectoryBuilder_h
#define RecoTracker_CkfPattern_TrackerTrajectoryBuilder_h

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/Event.h"

class Trajectory;
class TrajectorySeed;
class CkfDebugger;
class MeasurementTracker;
class Propagator;
class Chi2MeasurementEstimatorBase;
class TransientTrackingRecHitBuilder;


/** The component of track reconstruction that, strating from a seed,
 *  reconstructs all possible trajectories.
 *  The resulting trajectories may be mutually exclusive and require
 *  cleaning by a TrajectoryCleaner.
 *  The Trajectories are normally not smoothed.
 */

class TrackerTrajectoryBuilder {
public:

  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;

  virtual ~TrackerTrajectoryBuilder() {};

  virtual TrajectoryContainer trajectories(const TrajectorySeed&) const = 0;

  virtual void setEvent(const edm::Event& event) const = 0;

  virtual void setDebugger( CkfDebugger * dbg) const {;}

 protected:  
  //methods for dubugging 
  virtual bool analyzeMeasurementsDebugger(Trajectory& traj, std::vector<TrajectoryMeasurement> meas,
					   const MeasurementTracker* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual void fillSeedHistoDebugger(TrajectoryMeasurement result0, TrajectoryMeasurement result1) const {;}

};


#endif
