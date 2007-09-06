#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: 2007/02/01 17:57:59 $
 *  $Revision: 1.22 $
 *  \author R. Bellan - INFN Torino
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrajectoryBuilder;
class MuonTrajectoryCleaner;
class MuonTrackLoader;

class MuonTrackFinder {

  public:

    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;
    typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;  

  public:
  
    /// constructor, for the STA reconstruction the trackLoader must have the propagator.
    MuonTrackFinder(MuonTrajectoryBuilder* ConcreteMuonTrajectoryBuilder,
		    MuonTrackLoader *trackLoader);
    
    /// destructor
    virtual ~MuonTrackFinder();
  
    /// reconstruct standalone tracks starting from a collection of seeds
    edm::OrphanHandle<reco::TrackCollection> reconstruct(const edm::Handle<TrajectorySeedCollection>&,
							 edm::Event&);

    /// reconstruct global tracks starting from a collection of
    /// standalone tracks and one of trakectories. If the latter
    /// is invalid, trajectories are refitted.
    void reconstruct(const std::vector<TrackCand>&, edm::Event&);
    
 private:
    
    /// percolate the Event Setup
    void setEvent(const edm::Event&);

    /// convert the trajectories into tracks and load them in to the event
    edm::OrphanHandle<reco::TrackCollection> load(const TrajectoryContainer&, edm::Event&);

    /// convert the trajectories into tracks and load them in to the event
    void load(const CandidateContainer&, edm::Event&);

  private:

    MuonTrajectoryBuilder* theTrajBuilder;

    MuonTrajectoryCleaner* theTrajCleaner;

    MuonTrackLoader* theTrackLoader;
  
};
#endif 
