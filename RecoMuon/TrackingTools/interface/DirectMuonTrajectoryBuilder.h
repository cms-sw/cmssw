#ifndef RecoMuon_TrackingTools_DirectMuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_DirectMuonTrajectoryBuilder_H

/** \class DirectMuonTrajectoryBuilder
 *  Class which takes a trajectory seed and fit its hits, returning a Trajectory container
 *
 *  $Date: 2008/10/06 13:41:50 $
 *  $Revision: 1.2 $
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
  virtual ~DirectMuonTrajectoryBuilder();
  
    /// return a container of the reconstructed trajectories compatible with a given seed
  virtual TrajectoryContainer trajectories(const TrajectorySeed&);
  
  /// return a container reconstructed muons starting from a given track
  virtual CandidateContainer trajectories(const TrackCand&);
  
  /// pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);
  
 private:
     const MuonServiceProxy *theService;
     SeedTransformer* theSeedTransformer;
};
#endif
