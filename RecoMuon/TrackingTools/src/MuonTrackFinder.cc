/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

//FIXME??
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
//FIXME
#include "DataFormats/MuonReco/interface/RecoMuonCollection.h"



#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

using namespace std;

/// constructor
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder):
  theTrajBuilder(ConcreteMuonTrajectoryBuilder)
{
  theTrajCleaner = new MuonTrajectoryCleaner();
}

/// Destructor
MuonTrackFinder::~MuonTrackFinder(){
  delete theTrajBuilder;
  delete theTrajCleaner;
};

/// Reconstruct tray
auto_ptr<RecoMuonCollection> MuonTrackFinder::reconstruct(const edm::Handle<TrackingSeedCollection>& seeds, const edm::EventSetup& eSetup){

  // Traj container
  TrajectoryContainer muonTrajectories;

  // reconstruct the traj
  for(TrackingSeedCollection::const_iterator seed = seeds->begin();
      seed != seeds->end(); seed++){
    muonTrajectories = theTrajBuilder->trajectories(*seed);
  }

  // clean the clone traj
  theTrajCleaner->clean(muonTrajectories); //used by reference...
  return convert(muonTrajectories);
}
  

// Convert trajectory container in ...
auto_ptr<RecoMuonCollection> MuonTrackFinder::convert(TrajectoryContainer &cont) const{
  //...
  return auto_ptr<RecoMuonCollection>();
}



