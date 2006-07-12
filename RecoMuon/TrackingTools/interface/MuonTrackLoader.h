#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  $Date: 2006/07/06 09:19:05 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

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
  edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer &trajectories, 
						      edm::Event& event);
  
 protected:
  
 private:
  reco::Track buildTrack (const Trajectory& trajectory) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;


};
#endif

