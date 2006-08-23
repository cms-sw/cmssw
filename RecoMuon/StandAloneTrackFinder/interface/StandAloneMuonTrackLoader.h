#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonTrackLoader_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonTrackLoader_H

/** \class StandAloneMuonTrackLoader
 *  Concrete class to load the product of the StandAloneProducer in the event
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "DataFormats/TrackReco/interface/Track.h"

class StandAloneMuonTrackLoader: public MuonTrackLoader {
public:
  /// Constructor
  StandAloneMuonTrackLoader(){};

  /// Destructor
  virtual ~StandAloneMuonTrackLoader(){};

  // Operations

  /// Convert the trajectories in tracks and load the tracks in the event
  void loadTracks(const TrajectoryContainer& trajectories, 
		  edm::Event& event);
  
protected:

private:
  reco::Track buildTrack (const Trajectory& trajectory) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;

};
#endif

