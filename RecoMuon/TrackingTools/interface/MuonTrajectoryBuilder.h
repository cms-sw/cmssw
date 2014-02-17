#ifndef RecoMuon_TrackingTools_MuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_MuonTrajectoryBuilder_H

/** \class MuonTrajectoryBuilder
 *  Base class for the Muon reco Trajectory Builder 
 *
 *  $Date: 2008/02/04 14:58:52 $
 *  $Revision: 1.21 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include <vector>

namespace edm {class Event;}

class TrajectorySeed;

class MuonTrajectoryBuilder {

  public:
  
    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;
    typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

    /// constructor
    MuonTrajectoryBuilder() {}
  
    /// destructor
    virtual ~MuonTrajectoryBuilder() {}

    /// return a container of the reconstructed trajectories compatible with a given seed
    virtual TrajectoryContainer trajectories(const TrajectorySeed&) = 0;

    /// return a container reconstructed muons starting from a given track
    virtual CandidateContainer trajectories(const TrackCand&) = 0;

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event& event) = 0;
  
 private:
};
#endif
