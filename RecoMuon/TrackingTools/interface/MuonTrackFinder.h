#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: 2006/07/06 09:19:05 $
 *  $Revision: 1.10 $
 *  \author R. Bellan - INFN Torino
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
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
    void load(const TrajectoryContainer &trajectories, edm::Event &event);

  private:

    MuonTrajectoryBuilder* theTrajBuilder; // it is a base class

    MuonTrajectoryCleaner* theTrajCleaner;

    MuonTrackLoader *theTrackLoader;
  
};
#endif 
