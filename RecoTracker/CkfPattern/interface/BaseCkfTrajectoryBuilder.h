#ifndef RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h
#define RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<cassert>
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

class CkfDebugger;
class Chi2MeasurementEstimatorBase;
class DetGroup;
class FreeTrajectoryState;
class IntermediateTrajectoryCleaner;
class LayerMeasurements;
class MeasurementTracker;
class MeasurementEstimator;
class NavigationSchool;
class Propagator;
class TrajectoryStateUpdator;
class TrajectoryMeasurement;
class TrajectorySeed;
class TrajectoryContainer;
class TrajectoryStateOnSurface;
class TrajectoryFitter;
class TransientTrackingRecHitBuilder;
class Trajectory;
class TempTrajectory;
class TrajectoryFilter;
class TrackingRegion;
class TrajectoryMeasurementGroup;
class TrajectoryCleaner;

#include "TrackingTools/PatternTools/interface/bqueue.h"
#include "RecoTracker/CkfPattern/interface/PrintoutHelper.h"

/** The component of track reconstruction that, strating from a seed,
 *  reconstructs all possible trajectories.
 *  The resulting trajectories may be mutually exclusive and require
 *  cleaning by a TrajectoryCleaner.
 *  The Trajectories are normally not smoothed.
 */

class BaseCkfTrajectoryBuilder : public TrajectoryBuilder {
protected:
  // short names
  typedef FreeTrajectoryState         FTS;
  typedef TrajectoryStateOnSurface    TSOS;
  typedef TrajectoryMeasurement       TM;
  typedef std::pair<TSOS,std::vector<const DetLayer*> > StateAndLayers;

public:

  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef std::vector<TempTrajectory> TempTrajectoryContainer;
  typedef TrajectoryContainer::iterator TrajectoryIterator;
  
  BaseCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
			   const TrajectoryStateUpdator*         updator,
			   const Propagator*                     propagatorAlong,
			   const Propagator*                     propagatorOpposite,
			   const Chi2MeasurementEstimatorBase*   estimator,
			   const TransientTrackingRecHitBuilder* RecHitBuilder,
			   const MeasurementTracker*             measurementTracker,
			   const TrajectoryFilter*               filter,
			   const TrajectoryFilter*               inOutFilter = 0);

  virtual ~BaseCkfTrajectoryBuilder();

  // new interface returning the start Trajectory...
  virtual TempTrajectory buildTrajectories (const TrajectorySeed& seed,
					    TrajectoryContainer &ret,
					    const TrajectoryFilter*) const  { assert(0==1); return TempTrajectory();}
  
  
  virtual void  rebuildTrajectories(TempTrajectory const& startingTraj, const TrajectorySeed& seed,
				    TrajectoryContainer& result) const { assert(0==1);}


  virtual void setEvent(const edm::Event& event) const;
  virtual void unset() const;

  virtual void setDebugger( CkfDebugger * dbg) const {;}
 
  /** Maximum number of lost hits per trajectory candidate. */
  //  int 		maxLostHit()		{return theMaxLostHit;}

  /** Maximum number of consecutive lost hits per trajectory candidate. */
  //  int 		maxConsecLostHit()	{return theMaxConsecLostHit;}

 protected:    
  //methods for dubugging 
  virtual bool analyzeMeasurementsDebugger(Trajectory& traj, const std::vector<TrajectoryMeasurement>& meas,
					   const MeasurementTracker* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual bool analyzeMeasurementsDebugger(TempTrajectory& traj, const std::vector<TrajectoryMeasurement>& meas,
					   const MeasurementTracker* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual void fillSeedHistoDebugger(std::vector<TrajectoryMeasurement>::iterator begin, 
                                     std::vector<TrajectoryMeasurement>::iterator end) const {;}

 protected:

  TempTrajectory createStartingTrajectory( const TrajectorySeed& seed) const;

  /** Called after each new hit is added to the trajectory, to see if building this track should be continued */
  // If inOut is true, this is being called part-way through tracking, after the in-out tracking phase is complete.
  // If inOut is false, it is called at the end of tracking.
  bool toBeContinued( TempTrajectory& traj, bool inOut = false) const;

  /** Called at end of track building, to see if track should be kept */
  bool qualityFilter( const TempTrajectory& traj, bool inOut = false) const;
  
  void addToResult(boost::shared_ptr<const TrajectorySeed> const & seed, TempTrajectory& traj, TrajectoryContainer& result, bool inOut = false) const;    
  void addToResult( TempTrajectory const& traj, TempTrajectoryContainer& result, bool inOut = false) const;    
  void moveToResult( TempTrajectory&& traj, TempTrajectoryContainer& result, bool inOut = false) const;    

  StateAndLayers findStateAndLayers(const TrajectorySeed& seed, const TempTrajectory& traj) const;
  StateAndLayers findStateAndLayers(const TempTrajectory& traj) const;

 private:
  void seedMeasurements(const TrajectorySeed& seed, TempTrajectory & result) const;



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

 private:
  //  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  //  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
  //                                     per trajectory candidate. */
  //  int theMinimumNumberOfHits;   /**< Minimum number of hits for a trajectory to be returned.*/
  //  float theChargeSignificance;  /**< Value to declare (q/p)/sig(q/p) significant. Negative: ignore. */

  //  TrajectoryFilter*              theMinPtCondition;
  //  TrajectoryFilter*              theMaxHitsCondition;
  const TrajectoryFilter* theFilter; /** Filter used at end of complete tracking */
  const TrajectoryFilter* theInOutFilter; /** Filter used at end of in-out tracking */

  bool skipClusters_;
  edm::InputTag clustersToSkip_;
};


#endif
