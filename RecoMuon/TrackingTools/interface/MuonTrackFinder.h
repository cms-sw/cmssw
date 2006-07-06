#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: 2006/06/27 13:44:19 $
 *  $Revision: 1.9 $
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

class MuonTrackFinder{ 
  
 public:

  typedef std::vector<Trajectory> TrajectoryContainer;

 public:
  
  /// constructor
  MuonTrackFinder(MuonTrajectoryBuilder* ConcreteMuonTrajectoryBuilder); 
  
  /// Destructor
  virtual ~MuonTrackFinder();
  
  /// Reconstruct tracks
  void reconstruct(const edm::Handle<TrajectorySeedCollection>&,
		   edm::Event&,
		   const edm::EventSetup&);

 private:

  /// Percolate the Event Setup
  void setES(const edm::EventSetup&);

  /// Percolate the Event Setup
  void setEvent(const edm::Event&);

  /// Convert the trajectories into tracks and load them in to the event
  void load(const TrajectoryContainer &trajectories, edm::Event &event);

 private:

  MuonTrajectoryBuilder* theTrajBuilder; // It isn't the same as in ORCA!!Now it is a base class

  MuonTrajectoryCleaner* theTrajCleaner;

  MuonTrackLoader *theTrackLoader;

 protected:
  
};
#endif 

