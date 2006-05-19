#ifndef CkfTrajectoryBuilder_H
#define CkfTrajectoryBuilder_H

#include <vector>

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class TrajectorySeed;
class TrajectoryStateOnSurface;
class TrajectoryFilter;

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

using namespace std;

class CkfTrajectoryBuilder {
protected:
// short names
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  typedef std::vector<Trajectory>     TrajectoryContainer;

  CkfTrajectoryBuilder( const edm::ParameterSet& conf,
			const edm::EventSetup& es,
			const MeasurementTracker* theInputMeasurementTracker);

  ~CkfTrajectoryBuilder();
  
  /// trajectories building starting from a seed
  TrajectoryContainer trajectories(const TrajectorySeed& seed,edm::Event& e);

private:
  edm::ESHandle<TrajectoryStateUpdator>       theUpdator;
  edm::ESHandle<Propagator>                   thePropagator;
  edm::ESHandle<Propagator>                   thePropagatorOpposite;
  edm::ESHandle<Chi2MeasurementEstimatorBase> theEstimator;

  const MeasurementTracker*     theMeasurementTracker;
  const LayerMeasurements*      theLayerMeasurements;


  TrajectoryFilter*              theMinPtCondition;

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
