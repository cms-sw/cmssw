#ifndef HICTrajectoryBuilder_H
#define HICTrajectoryBuilder_H

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

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
//#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoHI/HiMuonAlgos/interface/HICMeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "RecoHI/HiMuonAlgos/interface/HICConst.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
class TransientTrackingRecHitBuilder;
class TrajectoryFilter;

class HICTrajectoryBuilder :public BaseCkfTrajectoryBuilder {
protected:
// short names
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:

  typedef std::vector<Trajectory>         TrajectoryContainer;
  typedef std::vector<TempTrajectory>     TempTrajectoryContainer;

  //HICTrajectoryBuilder( const edm::ParameterSet& conf,
  //			const edm::EventSetup& es,
  //		const MeasurementTracker* theInputMeasurementTracker);
  HICTrajectoryBuilder(const edm::ParameterSet&              conf,
                       const edm::EventSetup&                es, 
		       const TrajectoryStateUpdator*         updator,
		       const Propagator*                     propagatorAlong,
		       const Propagator*                     propagatorOpposite,
		       const Chi2MeasurementEstimatorBase*           estimator,
		       const TransientTrackingRecHitBuilder* RecHitBuilder,
		       const MeasurementTracker*             measurementTracker,
                       const TrajectoryFilter*               filter);

  ~HICTrajectoryBuilder();
  
  /// trajectories building starting from a seed
  virtual TrajectoryContainer trajectories(const TrajectorySeed& seed) const;

  /// set Event for the internal MeasurementTracker data member
  virtual void setEvent(const edm::Event& event) const;

  virtual void settracker(const MeasurementTracker* measurementTracker){theMeasurementTracker = measurementTracker;}

 private:
  const TrajectoryStateUpdator*         theUpdator;
  const Propagator*                     thePropagatorAlong;
  const Propagator*                     thePropagatorOpposite;
  const Chi2MeasurementEstimatorBase*   theEstimator;
//  const HICMeasurementEstimator*        theEstimator;
  mutable cms::HICConst*                theHICConst; 
    
  edm::ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  edm::ESHandle<TrajectoryFitter>       theFitterTrack;
  edm::ESHandle<TrajectorySmoother>     theSmootherTrack;
  edm::ESHandle<Propagator>             thePropagatorTrack;
  
  const TransientTrackingRecHitBuilder* theTTRHBuilder;
  const MeasurementTracker*             theMeasurementTracker;
  const LayerMeasurements*              theLayerMeasurements;

  // these may change from seed to seed
  mutable const Propagator*             theForwardPropagator;
  mutable const Propagator*             theBackwardPropagator;
  
  TrajectoryFilter*                     theMinPtCondition;
  TrajectoryFilter*                     theMaxHitsCondition;

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


  TempTrajectory createStartingTrajectory( const TrajectorySeed& seed) const;

  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;

  void limitedCandidates( TempTrajectory& startingTraj, TrajectoryContainer& result) const;

  std::vector<TrajectoryMeasurement> findCompatibleMeasurements( const TempTrajectory& traj) const;

  bool qualityFilter( const TempTrajectory& traj) const;

  void addToResult( TempTrajectory& traj, TrajectoryContainer& result) const; 
  
  bool updateTrajectory( TempTrajectory& traj, const TM& tm) const;

  bool toBeContinued( const TempTrajectory& traj) const;

};

#endif
