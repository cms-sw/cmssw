#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  $Date: 2006/06/27 13:44:19 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>

class Trajectory;
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
			  edm::Event& event);

 protected:
  
 private:
  reco::Track buildTrack (const Trajectory& trajectory) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;


};
#endif

