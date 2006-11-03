#ifndef RecoMuon_TrackingTools_MuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_MuonTrajectoryBuilder_H

/** \class MuonTrajectoryBuilder
 *  Base class for the Muon reco Trajectory Builder 
 *
 *  $Date: 2006/08/31 18:24:17 $
 *  $Revision: 1.13 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include <vector>

namespace edm {class EventSetup; class Event;}

class TrajectorySeed;

class MuonTrajectoryBuilder {

  public:
  
    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;
    typedef std::pair<Trajectory*, reco::TrackRef> TrackCand;

    /// constructor
    MuonTrajectoryBuilder() {}
  
    /// destructor
    virtual ~MuonTrajectoryBuilder() {}

    /// return a container of the reconstructed trajectories compatible with a given seed
    virtual TrajectoryContainer trajectories(const TrajectorySeed&) = 0;

    /// return a container reconstructed muons starting from a given track
    virtual CandidateContainer trajectories(const reco::TrackRef&) = 0;
    virtual CandidateContainer trajectories(const TrackCand&) = 0;

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event& event) = 0;
  
 private:

};
#endif
