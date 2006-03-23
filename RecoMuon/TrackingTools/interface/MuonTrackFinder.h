#ifndef RecoMuon_TrackingTools_MuonTrackFinder_H
#define RecoMuon_TrackingTools_MuonTrackFinder_H

/** \class MuonTrackFinder
 *  Track finder for the Muon Reco
 *
 *  $Date: 2006/03/21 13:29:48 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

//FIXME
#include "DataFormats/MuonReco/interface/RecoMuonCollection.h"

//FIXME??
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
//#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonTrajectoryBuilder;
class MuonTrajectoryCleaner;

class MuonTrackFinder{ 
  
 public:

  typedef std::vector<Trajectory> TrajectoryContainer;

 public:
  
  /// constructor
  MuonTrackFinder(MuonTrajectoryBuilder* ConcreteMuonTrajectoryBuilder);
  
  /// Destructor
  virtual ~MuonTrackFinder();
  
  /// Reconstruct tracks
  std::auto_ptr<RecoMuonCollection> reconstruct(const edm::Handle<TrajectorySeedCollection>&, const edm::EventSetup&);

 private:

  MuonTrajectoryBuilder* theTrajBuilder; // It isn't the same as in ORCA!!Now it is a base class

  MuonTrajectoryCleaner* theTrajCleaner;

  std::auto_ptr<RecoMuonCollection> convert(TrajectoryContainer&) const;
 protected:
  
};
#endif 

