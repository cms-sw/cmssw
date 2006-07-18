#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Base class to load the tracks in the event
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

class MuonTrackFinder;
namespace edm {class Event;}

class MuonTrackLoader {
public:
  typedef std::vector<Trajectory> TrajectoryContainer;
  
  /// Constructor
  MuonTrackLoader() {};

  /// Destructor
  virtual ~MuonTrackLoader(){};

  // Operations
  
  /// Convert the trajectories in tracks and load the tracks in the event
  virtual void loadTracks(const TrajectoryContainer &trajectories, 
			  edm::Event& event) = 0;

protected:

private:

};
#endif

