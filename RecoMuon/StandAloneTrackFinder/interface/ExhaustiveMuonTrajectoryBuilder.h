#ifndef ExhaustiveMuonTrajectoryBuilder_h
#define ExhaustiveMuonTrajectoryBuilder_h

/** Instead of letting the SeedGenerator code choose
    a segment to start from, this TrajectoryBuilder
    makes a seed wfor each segment, and chooses the
    Trajectory with the most hits and the lowest chi-squared
*/

#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonSeedFromRecHits.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

class ExhaustiveMuonTrajectoryBuilder : public MuonTrajectoryBuilder
{
public:
  ExhaustiveMuonTrajectoryBuilder(const edm::ParameterSet & pset, const MuonServiceProxy*,edm::ConsumesCollector& );
  ~ExhaustiveMuonTrajectoryBuilder() override;

  /// return a container of the reconstructed trajectories compatible with a given seed
  TrajectoryContainer trajectories(const TrajectorySeed&) override;

  /// return a container reconstructed muons starting from a given track
  CandidateContainer trajectories(const TrackCand&) override;

  /// pass the Event to the algo at each event
  void setEvent(const edm::Event& event) override;


private:
  void clean(TrajectoryContainer & trajectories) const;
  
  StandAloneMuonTrajectoryBuilder theTrajBuilder;
  MuonSeedFromRecHits theSeeder;
  const MuonServiceProxy *theService;

};

#endif

