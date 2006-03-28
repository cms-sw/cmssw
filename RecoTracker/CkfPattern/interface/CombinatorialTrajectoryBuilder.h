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
 
  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
                                     per trajectory candidate. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  bool theIntermediateCleaning;	/**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */
  int theMinHits;               /**< Minimum number of hits for a trajectory to be returned.*/
  bool theAlwaysUseInvalid;


  Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
  void limitedCandidates( Trajectory& startingTraj, TrajectoryContainer& result);
  std::vector<TrajectoryMeasurement> findCompatibleMeasurements( const Trajectory& traj);

  bool qualityFilter( const Trajectory& traj);
  void addToResult( Trajectory& traj, TrajectoryContainer& result);
  void updateTrajectory( Trajectory& traj, const TM& tm) const;
  bool toBeContinued( const Trajectory& traj);

};

#endif
