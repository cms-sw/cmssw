/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2006/07/31 11:10:39 $
 *  $Revision: 1.18 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"


using namespace std;

//
// constructor
//
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder) :
  theTrajBuilder(ConcreteMuonTrajectoryBuilder) {

  theTrackLoader = new MuonTrackLoader();
  theTrajCleaner = new MuonTrajectoryCleaner();

}


//
// destructor
//
MuonTrackFinder::~MuonTrackFinder() {

  LogDebug("Muon|RecoMuon|MuonTrackFinder")<<"MuonTrackFinder destructor called"<<endl;
  delete theTrajBuilder;
  delete theTrajCleaner;
  delete theTrackLoader;

}


//
// percolate the event setup
//
void MuonTrackFinder::setES(const edm::EventSetup& eSetup) {

  theTrajBuilder->setES(eSetup);
  theTrackLoader->setES(eSetup);

}


//
// percolate the event setup
//
void MuonTrackFinder::setEvent(const edm::Event& event) {

  theTrajBuilder->setEvent(event);

}


//
// convert the trajectories into tracks and load them in to the event
//
void MuonTrackFinder::load(const TrajectoryContainer& trajectories, 
			   edm::Event& event) {
                           
  theTrackLoader->loadTracks(trajectories, event);

}


//
// convert the trajectories into tracks and load them in to the event
//
void MuonTrackFinder::load(const CandidateContainer& muonCands,
			   edm::Event& event) {
                           
    theTrackLoader->loadTracks(muonCands, event);

}


//
// reconstruct trajectories
//
void MuonTrackFinder::reconstruct(const edm::Handle<TrajectorySeedCollection>& seeds,
				  edm::Event& event,
				  const edm::EventSetup& eSetup) {

  const std::string metname = "Muon|RecoMuon|MuonTrackFinder";
  
  // Percolate the event 
  LogDebug(metname)<<"Event setup percolation"<<endl;
  setEvent(event);

  // Percolate the event setup
  LogDebug(metname)<<"Event percolation"<<endl;
  setES(eSetup);
  
  // Trajectory container
  TrajectoryContainer muonTrajectories;
  
  // reconstruct the trajectory
  for(TrajectorySeedCollection::const_iterator seed = seeds->begin();
      seed != seeds->end(); seed++){
    LogDebug(metname)<<"+++ New Seed +++"<<endl;
    TrajectoryContainer muonTrajs_temp = theTrajBuilder->trajectories(*seed);
    for(TrajectoryContainer::const_iterator it = muonTrajs_temp.begin(); 
	it != muonTrajs_temp.end(); it++) 
      muonTrajectories.push_back(*it); 
  }

  // clean the clone traj
  LogDebug(metname)<<"Clean the trajectories container"<<endl;
  theTrajCleaner->clean(muonTrajectories); //used by reference...
  
  // convert the trajectories into tracks and load them in to the event
  LogDebug(metname)
    <<"Convert the trajectories into tracks and load them in to the event"<<endl;
  load(muonTrajectories,event);

}


//
// reconstruct trajectories
//
void MuonTrackFinder::reconstruct(const edm::Handle<reco::TrackCollection>& tracks,
				  edm::Event& event,
				  const edm::EventSetup& eSetup) {                       

  const std::string metname = "Muon|RecoMuon|MuonTrackFinder";

  // percolate the event 
  setEvent(event);

  // percolate the event setup
  setES(eSetup);
  
  // Muon Candidate container
  CandidateContainer muonCandidates;

  // 
  // reco::MuonCollection muonResult;

  // reconstruct the muon candidates
  for (unsigned int position = 0; position != tracks->size(); ++position) {
    LogDebug(metname)<<"+++ New Track +++"<<endl;
    reco::TrackRef staTrack(tracks,position);

    CandidateContainer muonCands_temp = theTrajBuilder->trajectories(staTrack);
    for (CandidateContainer::const_iterator it = muonCands_temp.begin(); it != muonCands_temp.end(); it++) {
      muonCandidates.push_back(*it); 
   }
  }                                  
  
  // clean the cloned candidates
  //IMPLEMENT ME

  // convert the trajectories into tracks and load them into the event
  LogDebug(metname)<<"Load Muon Candidates into the event"<<endl;
  //  load(muonCandidates,muonResult,event);            
  load(muonCandidates,event);

}
