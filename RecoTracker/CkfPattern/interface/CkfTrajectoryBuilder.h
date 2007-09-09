#ifndef CkfTrajectoryBuilder_H
#define CkfTrajectoryBuilder_H

#include <vector>

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class TrajectorySeed;
class TrajectoryStateOnSurface;
class TrajectoryFilter;

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"


#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"

class TransientTrackingRecHitBuilder;


class CkfTrajectoryBuilder :public TrackerTrajectoryBuilder {

public:

  typedef std::vector<Trajectory>         TrajectoryContainer;
  typedef std::vector<TempTrajectory>     TempTrajectoryContainer;

  CkfTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* recHitBuilder,
		       const MeasurementTracker*             measurementTracker);

  ~CkfTrajectoryBuilder() {}
  
  /// trajectories building starting from a seed
  virtual TrajectoryContainer trajectories(const TrajectorySeed& seed) const;

  /// set Event for the internal MeasurementTracker data member
  virtual void setEvent(const edm::Event& event) const;
  

 protected:
  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  bool theIntermediateCleaning;	/**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */
  bool theAlwaysUseInvalidHits;


 protected:
  virtual void findCompatibleMeasurements( const TempTrajectory& traj, std::vector<TrajectoryMeasurement> & result) const;

  void limitedCandidates( TempTrajectory& startingTraj, TrajectoryContainer& result) const;
  
  void updateTrajectory( TempTrajectory& traj, const TM& tm) const;

};

#endif
