#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

//FIXME
#include "DataFormats/MuonReco/interface/RecoMuonCollection.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrajectoryBuilder;

class MuonTrajectoryCleaner;

// FIXME the names
class SeedContainer;
class TrajectoryContainer;

//FIXME??
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"

class MuonTrackFinder{ 

  public:

  /// constructor
  MuonTrackFinder(MuonTrajectoryBuilder* ConcreteMuonTrajectoryBuilder);
  
  /// Destructor
  virtual ~MuonTrackFinder();
  
  /// Reconstruct tracks
  std::auto_ptr<RecoMuonCollection> reconstruct(const edm::Handle<TrackingSeedCollection>&, const edm::EventSetup&);

 private:

  MuonTrajectoryBuilder* theTrajBuilder; // It isn't the same as in ORCA!!Now it is a base class

  MuonTrajectoryCleaner* theTrajCleaner;

  std::auto_ptr<RecoMuonCollection> convert(TrajectoryContainer&) const;
 protected:
  
};
#endif 

