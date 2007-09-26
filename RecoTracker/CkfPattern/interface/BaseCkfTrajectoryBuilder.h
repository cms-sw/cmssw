#ifndef RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h
#define RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  typedef std::vector<Trajectory>     TrajectoryContainer;
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
			   const MeasurementTracker*             measurementTracker);

  virtual ~BaseCkfTrajectoryBuilder();

  virtual TrajectoryContainer trajectories(const TrajectorySeed&) const = 0;

  virtual void setEvent(const edm::Event& event) const = 0;

  virtual void setDebugger( CkfDebugger * dbg) const {;}
 
  /** Maximum number of lost hits per trajectory candidate. */
  int 		maxLostHit()		{return theMaxLostHit;}

  /** Maximum number of consecutive lost hits per trajectory candidate. */
  int 		maxConsecLostHit()	{return theMaxConsecLostHit;}

 protected:    
  //methods for dubugging 
  virtual bool analyzeMeasurementsDebugger(Trajectory& traj, std::vector<TrajectoryMeasurement> meas,
					   const MeasurementTracker* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual bool analyzeMeasurementsDebugger(TempTrajectory& traj, std::vector<TrajectoryMeasurement> meas,
					   const MeasurementTracker* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual void fillSeedHistoDebugger(std::vector<TrajectoryMeasurement>::iterator begin, 
                                     std::vector<TrajectoryMeasurement>::iterator end) const {;}

 protected:

  TempTrajectory createStartingTrajectory( const TrajectorySeed& seed) const;

  bool toBeContinued( const TempTrajectory& traj) const;

  bool qualityFilter( const TempTrajectory& traj) const;
  
  void addToResult( TempTrajectory& traj, TrajectoryContainer& result) const;    
 
  StateAndLayers findStateAndLayers(const TempTrajectory& traj) const;

 private:
  void seedMeasurements(const TrajectorySeed& seed, std::vector<TrajectoryMeasurement> & result) const;



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
  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
                                     per trajectory candidate. */
  int theMinimumNumberOfHits;   /**< Minimum number of hits for a trajectory to be returned.*/

  TrajectoryFilter*              theMinPtCondition;
  TrajectoryFilter*              theMaxHitsCondition;


};


#endif
