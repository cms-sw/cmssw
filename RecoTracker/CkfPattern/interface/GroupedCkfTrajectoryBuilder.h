#ifndef GroupedCkfTrajectoryBuilder_H
#define GroupedCkfTrajectoryBuilder_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include <vector>


#include "FWCore/Utilities/interface/Visibility.h"


/** A highly configurable trajectory builder that allows full
 *  exploration of the combinatorial tree of possible continuations,
 *  and provides efficient ways of trimming the combinatorial tree.
 */

class GroupedCkfTrajectoryBuilder : public BaseCkfTrajectoryBuilder {
  
 public:
  /// constructor from ParameterSet
  GroupedCkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);

  /// destructor
  virtual ~GroupedCkfTrajectoryBuilder(){}

  /// set Event for the internal MeasurementTracker data member
  //  virtual void setEvent(const edm::Event& event) const;

  /// trajectories building starting from a seed
  TrajectoryContainer trajectories(const TrajectorySeed&) const;

  /// trajectories building starting from a seed, return in an already allocated vector
  void trajectories(const TrajectorySeed&, TrajectoryContainer &ret) const;

  /// trajectories building starting from a seed with a region
  TrajectoryContainer trajectories(const TrajectorySeed&, const TrackingRegion&) const;

  /// trajectories building starting from a seed with a region
  void trajectories(const TrajectorySeed&, TrajectoryContainer &ret, const TrackingRegion&) const;

  /// common part of both public trajectory building methods
  // also new interface returning the start Trajectory...
  TempTrajectory buildTrajectories (const TrajectorySeed&seed,
				    TrajectoryContainer &ret,
				    const TrajectoryFilter*) const;



  /** trajectories re-building in the seeding region.
      It looks for additional measurements in the seeding region of the 
      intial trajectories.
      Only valid trajectories are returned. Invalid ones are dropped from the input
      collection.
  **/
  void  rebuildSeedingRegion(const TrajectorySeed&,
			     TrajectoryContainer& result) const ;
 
  // same as above using the precomputed startingTraj..
  void  rebuildTrajectories(TempTrajectory const & startingTraj, const TrajectorySeed&,
			     TrajectoryContainer& result) const ;  


  // Access to lower level components
  const TrajectoryStateUpdator&  updator() const    {return *theUpdator;}
  const Chi2MeasurementEstimatorBase&    estimator() const  {return *theEstimator;}

  //   PropagationDirection        direction() const  {return theDirection;}

  /** Chi**2 Cut on the new Trajectory Measurements to consider */
  double 	chiSquareCut()		{return theChiSquareCut;}

  /** Maximum number of trajectory candidates to propagate to the next layer. */
  int 		maxCand()		{return theMaxCand;}


  /** Chi**2 Penalty for each lost hit. */
  float 	lostHitPenalty()	{return theLostHitPenalty;}

  //   /** Tells whether an intermediary cleaning stage should take place during TB. */
  //   bool 		intermediateCleaning()	{return theIntermediateCleaning;}

  /// Pt cut
  double ptCut() {return theptCut;}

  /// Mass hypothesis used for propagation 
  double mass() {return theMass;}

protected:
  void setEvent_(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  virtual void analyseSeed(const TrajectorySeed& seed) const{}

  virtual void analyseMeasurements( const std::vector<TM>& meas, 
				    const Trajectory& traj) const{}
  virtual void analyseResult( const TrajectoryContainer& result) const {}

private :
//  /// no copy constructor
//  GroupedCkfTrajectoryBuilder (const GroupedCkfTrajectoryBuilder&)  = default;
//
//  /// no assignment operator
//  GroupedCkfTrajectoryBuilder& operator= (const GroupedCkfTrajectoryBuilder&)  dso_internal;

  
  inline bool tkxor(bool a, bool b) const  dso_internal {return (a||b) && !(a&&b);}
  // to be ported later

  bool advanceOneLayer( const TrajectorySeed& seed,
                        TempTrajectory& traj, 
			const TrajectoryFilter* regionalCondition,
			const Propagator* propagator, 
                        bool inOut,
			TempTrajectoryContainer& newCand, 
			TempTrajectoryContainer& result) const  dso_internal;

  void groupedLimitedCandidates( const TrajectorySeed& seed,
                                 TempTrajectory const& startingTraj, 
				 const TrajectoryFilter* regionalCondition,
				 const Propagator* propagator, 
                                 bool inOut,
				 TempTrajectoryContainer& result) const  dso_internal;

  /// try to find additional hits in seeding region
  void rebuildSeedingRegion (const TrajectorySeed&seed,
			     TempTrajectory const& startingTraj,
			     TempTrajectoryContainer& result) const  dso_internal;

   //** try to find additional hits in seeding region for a candidate
   //* (returns number of trajectories added) *
  int rebuildSeedingRegion (const TrajectorySeed&seed,
			    const std::vector<const TrackingRecHit*>& seedHits,
			    TempTrajectory& candidate,
			    TempTrajectoryContainer& result) const  dso_internal;

  // ** Backward fit of trajectory candidate except seed. Fit result is returned. invalid if fit failed
  // *  remaining hits are returned  remainingHits.
  TempTrajectory backwardFit (TempTrajectory& candidate, unsigned int nSeed,
			      const TrajectoryFitter& fitter,
			      std::vector<const TrackingRecHit*>& remainingHits) const  dso_internal;

  /// Verifies presence of a RecHits in a range of TrajectoryMeasurements.
  bool verifyHits (TempTrajectory::DataContainer::const_iterator rbegin,
		   size_t maxDepth,
		   const std::vector<const TrackingRecHit*>& hits) const  dso_internal;

  /// intermediate cleaning in the case of grouped measurements
  void groupedIntermediaryClean(TempTrajectoryContainer& theTrajectories) const  dso_internal;


  /// change of propagation direction
  static inline PropagationDirection oppositeDirection (PropagationDirection dir) {
    if ( dir==alongMomentum )  return oppositeToMomentum;
    if ( dir==oppositeToMomentum )  return alongMomentum;
    return dir;
  }


private:
  TrajectoryFilter*              theConfigurableCondition;

  //   typedef deque< const TrajectoryFilter*>   StopCondContainer;
  //   StopCondContainer              theStopConditions;

  double theChiSquareCut;       /**< Chi**2 Cut on the new Trajectory Measurements to consider */

  double theptCut;              /**< ptCut */

  double theMass;               /**< Mass hypothesis used for propagation */

  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  float theFoundHitBonus;       /**< Chi**2 bonus for each found hit (favours candidates with
				     more measurements) */
  bool theIntermediateCleaning;	/**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */

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
  bool theKeepOriginalIfRebuildFails;   
                                /**< Keep original trajectory if rebuilding fails. */

  /** If the value is greater than zero, the reconstructions for looper is turned on for
      candidates with pt greater than maxPtForLooperReconstruction */
  float maxPt2ForLooperReconstruction;

  float maxDPhiForLooperReconstruction;

  mutable TempTrajectoryContainer work_; // Better here than alloc every time
  enum work_MaxSize_Size_ { work_MaxSize_ = 50 };  // if it grows above this number, it is forced to resize to half this amount when cleared
};

#endif
