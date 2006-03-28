#ifndef CombinatorialTrajectoryBuilder_H
#define CombinatorialTrajectoryBuilder_H

#include <vector>

class MeasurementTracker;
class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class NavigationSchool;
class TrajectorySeed;
class Trajectory;
class TrajectoryStateOnSurface;
class TrajectoryMeasurement;

class CombinatorialTrajectoryBuilder {
protected:
// short names
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  typedef std::vector<Trajectory>     TrajectoryContainer;

  CombinatorialTrajectoryBuilder( const MeasurementTracker*,
				  const Propagator*,
				  const TrajectoryStateUpdator*,
				  const MeasurementEstimator*,
				  const NavigationSchool*);

  /// trajectories building starting from a seed
  TrajectoryContainer trajectories(const TrajectorySeed&);

private:

  const MeasurementTracker*     theTracker;
  Propagator*                   thePropagator;
  const TrajectoryStateUpdator* theUpdator;
  const MeasurementEstimator*   theEstimator;
  const NavigationSchool*       theNavigationSchool;
 
  Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
  void limitedCandidates( Trajectory& startingTraj, TrajectoryContainer& result);

};

#endif
