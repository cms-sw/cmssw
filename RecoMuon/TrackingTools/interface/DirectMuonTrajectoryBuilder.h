#ifndef RecoMuon_TrackingTools_DirectMuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_DirectMuonTrajectoryBuilder_H

/** \class DirectMuonTrajectoryBuilder
 *  Class which takes a trajectory seed and fit its hits, returning a Trajectory container
 *
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include <vector>

class MuonServiceProxy;
class SeedTransformer;
class TrajectorySeed;

namespace edm {class ParameterSet;}

class DirectMuonTrajectoryBuilder: public MuonTrajectoryBuilder {

 public:
  
    /// constructor
  DirectMuonTrajectoryBuilder(const edm::ParameterSet&, 
			      const MuonServiceProxy*);
  
  /// destructor
  ~DirectMuonTrajectoryBuilder() override;
  
    /// return a container of the reconstructed trajectories compatible with a given seed
  TrajectoryContainer trajectories(const TrajectorySeed&) override;
  
  /// return a container reconstructed muons starting from a given track
  CandidateContainer trajectories(const TrackCand&) override;
  
  /// pass the Event to the algo at each event
  void setEvent(const edm::Event& event) override;
  
 private:
     const MuonServiceProxy *theService;
     SeedTransformer* theSeedTransformer;
};
#endif
