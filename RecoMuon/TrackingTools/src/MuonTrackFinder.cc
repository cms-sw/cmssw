/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2006/06/14 17:47:07 $
 *  $Revision: 1.7 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

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

// percolate the event setup
void MuonTrackFinder::setEvent(const edm::Event &event){
  theTrajBuilder->setEvent(event);
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
  

auto_ptr<reco::TrackCollection> MuonTrackFinder::convert(TrajectoryContainer &cont) const{
  return auto_ptr<reco::TrackCollection>();
}
