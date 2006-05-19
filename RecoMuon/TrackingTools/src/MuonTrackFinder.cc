/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2006/05/17 13:05:15 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"


#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

using namespace std;

// constructor
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder):
  theTrajBuilder(ConcreteMuonTrajectoryBuilder)
{
  theTrajCleaner = new MuonTrajectoryCleaner();
}

// Destructor
MuonTrackFinder::~MuonTrackFinder(){
  delete theTrajBuilder;
  delete theTrajCleaner;
};

// percolate the event setup
void MuonTrackFinder::setES(const edm::EventSetup &eSetup){
  theTrajBuilder->setES(eSetup);
}

// Reconstruct trajectories
auto_ptr<reco::TrackCollection> MuonTrackFinder::reconstruct(const edm::Handle<TrajectorySeedCollection>& seeds){

  // Traj container
  TrajectoryContainer muonTrajectories;
  
  // reconstruct the traj
  for(TrajectorySeedCollection::const_iterator seed = seeds->begin();
      seed != seeds->end(); seed++){
    TrajectoryContainer muonTrajs_temp = theTrajBuilder->trajectories(*seed);
    for(TrajectoryContainer::const_iterator it = muonTrajs_temp.begin(); 
	it != muonTrajs_temp.end(); it++) 
      muonTrajectories.push_back(*it); 
  }

  // clean the clone traj
  theTrajCleaner->clean(muonTrajectories); //used by reference...
  return convert(muonTrajectories);
}
  

// Convert trajectory container in ...
auto_ptr<reco::TrackCollection> MuonTrackFinder::convert(TrajectoryContainer &cont) const{
  //...
  return auto_ptr<reco::TrackCollection>();
}



