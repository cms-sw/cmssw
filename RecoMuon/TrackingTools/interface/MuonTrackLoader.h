#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  $Date: 2006/10/24 09:20:06 $
 *  $Revision: 1.11 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace edm {class Event; class EventSetup; class ParameterSet;}

class Trajectory;
class Propagator;
class MuonServiceProxy;

class MuonTrackLoader {
  public:

    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;

    /// Constructor for the STA reco the args must be specify!
    MuonTrackLoader(edm::ParameterSet &parameterSet, const MuonServiceProxy *service =0);

    /// Destructor
    virtual ~MuonTrackLoader() {}
  
    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&);

    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&,const std::string&);

    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::MuonCollection> loadTracks(const CandidateContainer&,
                                                       edm::Event&); 
  
  private:
 
    reco::Track buildTrack (const Trajectory&) const;
    reco::TrackExtra buildTrackExtra(const Trajectory&) const;

    std::string thePropagatorName;
    bool theTrajectoryFlag;
    const MuonServiceProxy *theService;
};
#endif
