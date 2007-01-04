/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2006/11/23 02:29:10 $
 *  $Revision: 1.29 $
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

#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

using namespace std;
using namespace edm;

// constructor. For the STA reconstruction the trackLoader must have the propagator.
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder,
				 MuonTrackLoader *trackLoader) :
  theTrajBuilder(ConcreteMuonTrajectoryBuilder),
  theTrackLoader(trackLoader){
  
  theTrajCleaner = new MuonTrajectoryCleaner();

}


// destructor
MuonTrackFinder::~MuonTrackFinder() {

  LogDebug("Muon|RecoMuon|MuonTrackFinder")<<"MuonTrackFinder destructor called"<<endl;
  delete theTrajBuilder;
  delete theTrajCleaner;
  delete theTrackLoader;

}

// percolate the event setup
void MuonTrackFinder::setEvent(Event& event) {
  theTrajBuilder->setEvent(event);
}

// convert the trajectories into tracks and load them in to the event
void MuonTrackFinder::load(const TrajectoryContainer& trajectories, 
			   Event& event) {

  theTrackLoader->loadTracks(trajectories, event);
}

// convert the trajectories into tracks and load them in to the event
void MuonTrackFinder::load(const CandidateContainer& muonCands,
			   Event& event) {
                           
    theTrackLoader->loadTracks(muonCands, event);

}

// reconstruct trajectories
void MuonTrackFinder::reconstruct(const Handle<TrajectorySeedCollection>& seeds,
				  Event& event){

  const string metname = "Muon|RecoMuon|MuonTrackFinder";
  
  // Percolate the event 
  LogDebug(metname)<<"Event percolation"<<endl;  
  setEvent(event);
  
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


// reconstruct trajectories
void MuonTrackFinder::reconstruct(const Handle<reco::TrackCollection>& staTracks,
				  const Handle<vector<Trajectory> >& staTrajs,
				  Event& event){

  const string metname = "Muon|RecoMuon|MuonTrackFinder";

  typedef MuonTrajectoryBuilder::TrackCand TrackCand;

  // percolate the event 
  setEvent(event);

  // Muon Candidate container
  CandidateContainer muonCandidates;


  const vector<Trajectory>* trajCollection = 0;

  bool validTrajs = staTrajs.isValid();
  if ( validTrajs && staTrajs->size()!=staTracks->size()){
    LogError(metname) << "MuonTrackFinder::reconstruct: Size of trajectory and track collections do not match";
    validTrajs=false;
  } 

  if (validTrajs) {
    trajCollection = staTrajs.product();
  }

  // reconstruct the muon candidates
  for (unsigned int position = 0; position != staTracks->size(); ++position) {
    LogDebug(metname)<<"+++ New Track +++"<<endl;
    reco::TrackRef staTrack(staTracks,position);

    TrackCand staCand(0,staTrack);

    if (validTrajs) {
      vector<Trajectory>::const_iterator it = trajCollection->begin()+position;
      const Trajectory* trajRef(&*it);  
      if ( trajRef->isValid() ) staCand.first = trajRef;
    }

    CandidateContainer muonCands_temp = theTrajBuilder->trajectories(staCand);
    muonCandidates.insert(muonCandidates.end(), muonCands_temp.begin(),muonCands_temp.end());
  }                                  
  
  // clean the cloned candidates
  theTrajCleaner->clean(muonCandidates);

  // convert the trajectories into staTracks and load them into the event
  LogDebug(metname)<<"Load Muon Candidates into the event"<<endl;
  load(muonCandidates,event);

}
