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
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"

class TransientTrackingRecHitBuilder;


class CkfTrajectoryBuilder :public TrackerTrajectoryBuilder {
protected:
// short names
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  typedef std::vector<Trajectory>         TrajectoryContainer;
  typedef std::vector<TempTrajectory>     TempTrajectoryContainer;

  //CkfTrajectoryBuilder( const edm::ParameterSet& conf,
  //			const edm::EventSetup& es,
  //		const MeasurementTracker* theInputMeasurementTracker);
  CkfTrajectoryBuilder(const edm::ParameterSet&              conf,
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*   estimator,
		       const TransientTrackingRecHitBuilder* RecHitBuilder,
		       const MeasurementTracker*             measurementTracker);

  ~CkfTrajectoryBuilder();
  
  /// trajectories building starting from a seed
  virtual TrajectoryContainer trajectories(const TrajectorySeed& seed) const;

  /// set Event for the internal MeasurementTracker data member
  virtual void setEvent(const edm::Event& event) const;


 protected:
  const TrajectoryStateUpdator*         theUpdator;
  const Propagator*                     thePropagatorAlong;
  const Propagator*                     thePropagatorOpposite;
  const Chi2MeasurementEstimatorBase*   theEstimator;
  const TransientTrackingRecHitBuilder* theTTRHBuilder;
  const MeasurementTracker*             theMeasurementTracker;
  const LayerMeasurements*              theLayerMeasurements;

  // these may change from seed to seed
  mutable const Propagator*             theForwardPropagator;
  mutable const Propagator*             theBackwardPropagator;
  
  TrajectoryFilter*              theMinPtCondition;
  TrajectoryFilter*              theMaxHitsCondition;


 protected:
  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
                                     per trajectory candidate. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  bool theIntermediateCleaning;	/**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */
  int theMinimumNumberOfHits;   /**< Minimum number of hits for a trajectory to be returned.*/
  bool theAlwaysUseInvalidHits;


 protected:
  virtual void findCompatibleMeasurements( const TempTrajectory& traj, std::vector<TrajectoryMeasurement> & result) const;

  TempTrajectory createStartingTrajectory( const TrajectorySeed& seed) const;

  void seedMeasurements(const TrajectorySeed& seed, std::vector<TrajectoryMeasurement> & result) const;

  void limitedCandidates( TempTrajectory& startingTraj, TrajectoryContainer& result) const;

  bool qualityFilter( const TempTrajectory& traj) const;

  void addToResult( TempTrajectory& traj, TrajectoryContainer& result) const; 
  
  void updateTrajectory( TempTrajectory& traj, const TM& tm) const;

  bool toBeContinued( const TempTrajectory& traj) const;

};

#endif
