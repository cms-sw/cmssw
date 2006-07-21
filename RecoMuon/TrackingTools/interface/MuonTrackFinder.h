#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: 2006/07/20 15:41:32 $
 *  $Revision: 1.13 $
 *  \author R. Bellan - INFN Torino
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrajectoryBuilder;
class MuonTrajectoryCleaner;
class MuonTrackLoader;

class MuonTrackFinder { 
  
  public:

    typedef std::vector<Trajectory> TrajectoryContainer;
    typedef std::pair<Trajectory, reco::TrackRef> MuonCandidate; 
    typedef std::vector<MuonCandidate> CandidateContainer;

  public:
  
    /// constructor
    MuonTrackFinder(MuonTrajectoryBuilder* ConcreteMuonTrajectoryBuilder); 
  
    /// destructor
    virtual ~MuonTrackFinder();
  
    /// reconstruct tracks
    void reconstruct(const edm::Handle<TrajectorySeedCollection>&,
                     edm::Event&,
                     const edm::EventSetup&);
                     
    void reconstruct(const edm::Handle<reco::TrackCollection>&,  
                     edm::Event&,
                     const edm::EventSetup&);            

  private:

    /// percolate the Event Setup
    void setES(const edm::EventSetup&);

    /// percolate the Event Setup
    void setEvent(const edm::Event&);

    /// convert the trajectories into tracks and load them in to the event
    void load(const TrajectoryContainer&, edm::Event&);

    /// convert the trajectories into tracks and load them in to the event
    void load(const CandidateContainer&, const reco::MuonCollection&, edm::Event&);

  private:

    MuonTrajectoryBuilder* theTrajBuilder; // it is a base class

    MuonTrajectoryCleaner* theTrajCleaner;

    MuonTrackLoader* theTrackLoader;
  
};
#endif 
