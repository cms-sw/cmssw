#ifndef RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h
#define RecoTracker_CkfPattern_BaseCkfTrajectoryBuilder_h

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include<cassert>
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

class CkfDebugger;
class Chi2MeasurementEstimatorBase;
class DetGroup;
class FreeTrajectoryState;
class IntermediateTrajectoryCleaner;
class LayerMeasurements;
class MeasurementTracker;
class MeasurementTrackerEvent;
class MeasurementEstimator;
class NavigationSchool;
class Propagator;
class TrajectoryStateUpdator;
class TrajectoryMeasurement;
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
class TrackingComponentsRecord;
namespace edm {
  class ConsumesCollector;
}

#include "TrackingTools/PatternTools/interface/bqueue.h"
#include "RecoTracker/CkfPattern/interface/PrintoutHelper.h"

#include <string>

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

  // Claims ownership of TrajectoryFilter pointers
  BaseCkfTrajectoryBuilder(const edm::ParameterSet& conf,
                           TrajectoryFilter *filter,
                           TrajectoryFilter *inOutFilter=nullptr);
  BaseCkfTrajectoryBuilder(const BaseCkfTrajectoryBuilder &) = delete;
  BaseCkfTrajectoryBuilder& operator=(const BaseCkfTrajectoryBuilder&) = delete;
  virtual ~BaseCkfTrajectoryBuilder();

  // new interface returning the start Trajectory...
  virtual TempTrajectory buildTrajectories (const TrajectorySeed& seed,
					    TrajectoryContainer &ret,
					    const TrajectoryFilter*) const  { assert(0==1); return TempTrajectory();}
  
  
  virtual void  rebuildTrajectories(TempTrajectory const& startingTraj, const TrajectorySeed& seed,
				    TrajectoryContainer& result) const { assert(0==1);}


  virtual void setEvent(const edm::Event& event) const ;
  virtual void unset() const;

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const MeasurementTrackerEvent *data);

  virtual void setDebugger( CkfDebugger * dbg) const {;}
 
  /** Maximum number of lost hits per trajectory candidate. */
  //  int 		maxLostHit()		{return theMaxLostHit;}

  /** Maximum number of consecutive lost hits per trajectory candidate. */
  //  int 		maxConsecLostHit()	{return theMaxConsecLostHit;}


  const TransientTrackingRecHitBuilder* hitBuilder() const { return theTTRHBuilder;}

 protected:    
  static TrajectoryFilter *createTrajectoryFilter(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);

  virtual void setEvent_(const edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  //methods for dubugging 
  virtual bool analyzeMeasurementsDebugger(Trajectory& traj, const std::vector<TrajectoryMeasurement>& meas,
					   const MeasurementTrackerEvent* theMeasurementTracker, 
					   const Propagator* theForwardPropagator, 
					   const Chi2MeasurementEstimatorBase* theEstimator, 
					   const TransientTrackingRecHitBuilder * theTTRHBuilder) const {return true;} 
  virtual bool analyzeMeasurementsDebugger(TempTrajectory& traj, const std::vector<TrajectoryMeasurement>& meas,
					   const MeasurementTrackerEvent* theMeasurementTracker, 
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
  void setData(const MeasurementTrackerEvent *data) ;

  const Propagator *forwardPropagator(const TrajectorySeed& seed) const {
    return seed.direction() == alongMomentum ? thePropagatorAlong : thePropagatorOpposite;
  }
  const Propagator *backwardPropagator(const TrajectorySeed& seed) const {
    return seed.direction() == alongMomentum ? thePropagatorOpposite : thePropagatorAlong;
  }

 protected:
  typedef TrackingComponentsRecord Chi2MeasurementEstimatorRecord;

  const TrajectoryStateUpdator*         theUpdator;
  const Propagator*                     thePropagatorAlong;
  const Propagator*                     thePropagatorOpposite;
  const Chi2MeasurementEstimatorBase*   theEstimator;
  const TransientTrackingRecHitBuilder* theTTRHBuilder;
  const MeasurementTrackerEvent*        theMeasurementTracker;

 private:
  //  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  //  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
  //                                     per trajectory candidate. */
  //  int theMinimumNumberOfHits;   /**< Minimum number of hits for a trajectory to be returned.*/
  //  float theChargeSignificance;  /**< Value to declare (q/p)/sig(q/p) significant. Negative: ignore. */

  //  TrajectoryFilter*              theMinPtCondition;
  //  TrajectoryFilter*              theMaxHitsCondition;
  std::unique_ptr<TrajectoryFilter> theFilter; /** Filter used at end of complete tracking */
  std::unique_ptr<TrajectoryFilter> theInOutFilter; /** Filter used at end of in-out tracking */

  // for EventSetup
  const std::string theUpdatorName;
  const std::string thePropagatorAlongName;
  const std::string thePropagatorOppositeName;
  const std::string theEstimatorName;
  const std::string theRecHitBuilderName;
};


#endif
