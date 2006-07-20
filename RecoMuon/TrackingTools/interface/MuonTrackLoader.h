#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  $Date: 2006/07/12 16:33:04 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

#include <vector>

namespace edm {class Event;}

class Trajectory;

class MuonTrackLoader {

  public:

    typedef std::vector<Trajectory> TrajectoryContainer;
    typedef std::pair<Trajectory, reco::TrackRef> MuonCandidate;
    typedef std::vector<MuonCandidate> CandidateContainer;

  public:
    
    /// Constructor
    MuonTrackLoader() {}

    /// Destructor
    virtual ~MuonTrackLoader() {}
  
    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&);

    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::MuonCollection> loadTracks(const CandidateContainer&, 
                                                       edm::Event&); 
  
  private:
 
    reco::Track buildTrack (const Trajectory&) const;
    reco::TrackExtra buildTrackExtra(const Trajectory&) const;

};
#endif
