#ifndef GroupedCkfTrajectoryBuilder_H
#define GroupedCkfTrajectoryBuilder_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include <vector>

class Propagator;
class TrajectoryStateUpdator;
class Chi2MeasurementEstimatorBase;
class MeasurementEstimator;
class NavigationSchool;
class Trajectory;
class TrajectorySeed;
class TrajectoryContainer;
class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class TrajectoryMeasurement;
class TrajectoryFilter;
class TrackingRegion;
class TrajectoryMeasurementGroup;
class MeasurementTracker;
class LayerMeasurements;
class DetGroup;
//B.M. class RecHitEqualByChannels;
class TrajectoryFitter;
class TransientTrackingRecHitBuilder;

/** A highly configurable trajectory builder that allows full
 *  exploration of the combinatorial tree of possible continuations,
 *  and provides efficient ways of trimming the combinatorial tree.
 */

class GroupedCkfTrajectoryBuilder : public TrackerTrajectoryBuilder {

 protected:
  // short names
  typedef FreeTrajectoryState         FTS;
  typedef TrajectoryStateOnSurface    TSOS;
  typedef TrajectoryMeasurement       TM;
  typedef std::vector<Trajectory>     TrajectoryContainer;
  
 public:
  /// constructor from ParameterSet
  GroupedCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
			      const TrajectoryStateUpdator*         updator,
			      const Propagator*                     propagatorAlong,
			      const Propagator*                     propagatorOpposite,
			      const Chi2MeasurementEstimatorBase*   estimator,
			      const TransientTrackingRecHitBuilder* RecHitBuilder,
			      const MeasurementTracker*             measurementTracker);

  /// destructor
  virtual ~GroupedCkfTrajectoryBuilder();

  /// set Event for the internal MeasurementTracker data member
  virtual void setEvent(const edm::Event& event) const;

  /// trajectories building starting from a seed
  TrajectoryContainer trajectories(const TrajectorySeed&) const;

  /// trajectories building starting from a seed with a region
  TrajectoryContainer trajectories(const TrajectorySeed&, const TrackingRegion&) const;

  // Access to lower level components
  //B.M. const Propagator&           propagator() const {return *thePropagator;}
  const TrajectoryStateUpdator&  updator() const    {return *theUpdator;}
  const Chi2MeasurementEstimatorBase&    estimator() const  {return *theEstimator;}

  //   PropagationDirection        direction() const  {return theDirection;}

  /** Chi**2 Cut on the new Trajectory Measurements to consider */
  double 	chiSquareCut()		{return theChiSquareCut;}

  /** Maximum number of trajectory candidates to propagate to the next layer. */
  int 		maxCand()		{return theMaxCand;}

  /** Maximum number of lost hits per trajectory candidate. */
  int 		maxLostHit()		{return theMaxLostHit;}

  /** Maximum number of consecutive lost hits per trajectory candidate. */
  int 		maxConsecLostHit()	{return theMaxConsecLostHit;}

  /** Chi**2 Penalty for each lost hit. */
  float 	lostHitPenalty()	{return theLostHitPenalty;}

  //   /** Tells whether an intermediary cleaning stage should take place during TB. */
  //   bool 		intermediateCleaning()	{return theIntermediateCleaning;}

  /// Pt cut
  double ptCut() {return theptCut;}

  /// Mass hypothesis used for propagation 
  double mass() {return theMass;}

protected:

  virtual void analyseSeed(const TrajectorySeed& seed) const{}

  virtual void analyseMeasurements( const std::vector<TM>& meas, 
				    const Trajectory& traj) const{}
  virtual void analyseResult( const TrajectoryContainer& result) const {}

private :
  /// no copy constructor
  GroupedCkfTrajectoryBuilder (const GroupedCkfTrajectoryBuilder&);

  /// no assignment operator
  GroupedCkfTrajectoryBuilder& operator= (const GroupedCkfTrajectoryBuilder&);

  /// common part of both public trajectory building methods
  TrajectoryContainer buildTrajectories (const TrajectorySeed&,
					 const TrajectoryFilter*) const;

  Trajectory createStartingTrajectory( const TrajectorySeed&) const;

  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;

  void addToResult( Trajectory& traj, TrajectoryContainer& result) const;
  
  bool qualityFilter( const Trajectory& traj) const;
  bool toBeContinued( const Trajectory& traj, const TrajectoryFilter* regionalCondition) const;

  //B.M.TrajectoryContainer intermediaryClean(TrajectoryContainer& theTrajectories);
  // to be ported later

  //B.M. inline bool tkxor(bool a, bool b) const {return (a||b) && !(a&&b);}
  // to be ported later

  bool advanceOneLayer( Trajectory& traj, 
			const TrajectoryFilter* regionalCondition, 
			TrajectoryContainer& newCand, 
			TrajectoryContainer& result) const;

  void groupedLimitedCandidates( Trajectory& startingTraj, 
				 const TrajectoryFilter* regionalCondition, 
				 TrajectoryContainer& result) const;

  /* ======= B.M.to be ported later =================
  /// try to find additional hits in seeding region
  void rebuildSeedingRegion (Trajectory& startingTraj,
			     TrajectoryContainer& result);

   ** try to find additional hits in seeding region for a candidate
   * (returns number of trajectories added) *
  int rebuildSeedingRegion (const std::vector<RecHit>& seedHits,
			    Trajectory& candidate,
			    TrajectoryContainer& result);

   ** Backward fit of trajectory candidate except seed. Fit result and
   *  remaining hits are returned in fittedTracks and remainingHits.
   *  FittedTracks is empty if no fit was done. *
  void backwardFit (Trajectory& candidate, unsigned int nSeed,
		    const TrajectoryFitter& fitter,
		    TrajectoryContainer& fittedTracks,
		    std::vector<RecHit>& remainingHits) const;

  /// Verifies presence of a RecHits in a range of TrajectoryMeasurements.
  bool verifyHits (std::vector<TM>::const_iterator tmBegin,
		   std::vector<TM>::const_iterator tmEnd,
		   const RecHitEqualByChannels& recHitEqual,
		   const std::vector<RecHit>& hits) const;

  /// intermediate cleaning in the case of grouped measurements
  TrajectoryContainer groupedIntermediaryClean(TrajectoryContainer& theTrajectories);

  /// list of layers from a container of TrajectoryMeasurements
  std::vector<const DetLayer*> layers (const std::vector<TM>& measurements) const;

  /// change of propagation direction
  inline PropagationDirection oppositeDirection (PropagationDirection dir) const {
    if ( dir==alongMomentum )  return oppositeToMomentum;
    else if ( dir==oppositeToMomentum )  return alongMomentum;
    return dir;
  }

  ======================================================== */

private:
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
  TrajectoryFilter*              theConfigurableCondition;

  //   typedef deque< const TrajectoryFilter*>   StopCondContainer;
  //   StopCondContainer              theStopConditions;

  double theChiSquareCut;       /**< Chi**2 Cut on the new Trajectory Measurements to consider */

  double theptCut;              /**< ptCut */

  double theMass;               /**< Mass hypothesis used for propagation */

  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  int theMaxLostHit;            /**< Maximum number of lost hits per trajectory candidate.*/
  int theMaxConsecLostHit;      /**< Maximum number of consecutive lost hits 
                                     per trajectory candidate. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  float theFoundHitBonus;       /**< Chi**2 bonus for each found hit (favours candidates with
				     more measurements) */
  bool theIntermediateCleaning;	/**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */
  int theMinHits;               /**< Minimum number of hits for a trajectory to be returned.*/
  bool theAlwaysUseInvalid;

  bool theLockHits;             /**< Lock hits when building segments in a layer */
  bool theBestHitOnly;          /**< Use only best hit / group when building segments */

  bool theRequireSeedHitsInRebuild; 
                               /**< Only accept rebuilt trajectories if they contain the seed hits. */
  unsigned int theMinNrOfHitsForRebuild;     
                                /**< Minimum nr. of non-seed hits required for rebuild. 
                                     If ==0 the seeding part will remain untouched. */
  unsigned int theMinNrOf2dHitsForRebuild;   
                                /**< Minimum nr. of non-seed 2D hits required for rebuild. */
};

#endif
