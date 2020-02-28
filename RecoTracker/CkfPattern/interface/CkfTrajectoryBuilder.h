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

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include <map>
#include <boost/unordered_map.hpp>

class TransientTrackingRecHitBuilder;
class TrajectoryFilter;

class CkfTrajectoryBuilder : public BaseCkfTrajectoryBuilder {
public:
  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef std::vector<TempTrajectory> TempTrajectoryContainer;

  CkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  CkfTrajectoryBuilder(const edm::ParameterSet& conf, std::unique_ptr<TrajectoryFilter> filter);

  ~CkfTrajectoryBuilder() override {}

  /// trajectories building starting from a seed
  TrajectoryContainer trajectories(const TrajectorySeed& seed) const override;
  /// trajectories building starting from a seed
  void trajectories(const TrajectorySeed& seed, TrajectoryContainer& ret) const override;

  // new interface returning the start Trajectory...
  TempTrajectory buildTrajectories(const TrajectorySeed&,
                                   TrajectoryContainer& ret,
                                   unsigned int& nCandPerSeed,
                                   const TrajectoryFilter*) const override;

  void rebuildTrajectories(TempTrajectory const& startingTraj,
                           const TrajectorySeed&,
                           TrajectoryContainer& result) const override {}

  /// set Event for the internal MeasurementTracker data member
  //  virtual void setEvent(const edm::Event& event) const;

protected:
  int theMaxCand;               /**< Maximum number of trajectory candidates 
		                     to propagate to the next layer. */
  float theLostHitPenalty;      /**< Chi**2 Penalty for each lost hit. */
  bool theIntermediateCleaning; /**< Tells whether an intermediary cleaning stage 
                                     should take place during TB. */
  bool theAlwaysUseInvalidHits;

protected:
  void setEvent_(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  virtual void findCompatibleMeasurements(const TrajectorySeed& seed,
                                          const TempTrajectory& traj,
                                          std::vector<TrajectoryMeasurement>& result) const;

  unsigned int limitedCandidates(const TrajectorySeed& seed,
                                 TempTrajectory& startingTraj,
                                 TrajectoryContainer& result) const;
  unsigned int limitedCandidates(const std::shared_ptr<const TrajectorySeed>& sharedSeed,
                                 TempTrajectoryContainer& candidates,
                                 TrajectoryContainer& result) const;

  void updateTrajectory(TempTrajectory& traj, TM&& tm) const;

  /*  
      //not mature for integration.  
      bool theSharedSeedCheck;
      std::string theUniqueName;
      void rememberSeedAndTrajectories(const TrajectorySeed& seed,TrajectoryContainer &result) const;
      bool seedAlreadyUsed(const TrajectorySeed& seed,TempTrajectoryContainer &candidates) const;
      bool sharedSeed(const TrajectorySeed& seed1,const TrajectorySeed& seed2) const;
      //  mutable TempTrajectoryContainer theCachedTrajectories;
      typedef boost::unordered_multimap<uint32_t,TempTrajectory> SharedTrajectory;
      mutable SharedTrajectory theCachedTrajectories;
  */
};

#endif
