#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  $Date: 2006/07/31 11:09:34 $
 *  $Revision: 1.7 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace edm {class Event; class EventSetup;}

class Trajectory;
class Propagator;
class MuonServiceProxy;

class MuonTrackLoader {
  public:

    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;

    /// Constructor for the STA reco the args must be specify!
    MuonTrackLoader(std::string trackLoaderPropagatorName =0, const MuonServiceProxy *service =0);
    
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

    std::string thePropagatorName;
    const MuonServiceProxy *theService;
};
#endif
