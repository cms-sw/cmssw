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

class ExhaustiveMuonTrajectoryBuilder : public MuonTrajectoryBuilder
{
public:
  ExhaustiveMuonTrajectoryBuilder(const edm::ParameterSet & pset, const MuonServiceProxy*);
  virtual ~ExhaustiveMuonTrajectoryBuilder();

  /// return a container of the reconstructed trajectories compatible with a given seed
  virtual TrajectoryContainer trajectories(const TrajectorySeed&);

  /// return a container reconstructed muons starting from a given track
  virtual CandidateContainer trajectories(const TrackCand&);

  /// pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);


private:
  void clean(TrajectoryContainer & trajectories) const;
  
  StandAloneMuonTrajectoryBuilder theTrajBuilder;
  MuonSeedFromRecHits theSeeder;
  const MuonServiceProxy *theService;

};

#endif

