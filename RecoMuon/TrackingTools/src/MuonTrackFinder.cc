/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2010/03/24 19:50:19 $
 *  $Revision: 1.40 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
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

// Constructor, with default cleaner. For the STA reconstruction the trackLoader must have the propagator.
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder,
				 MuonTrackLoader *trackLoader) :
  theTrajBuilder(ConcreteMuonTrajectoryBuilder),
  theTrajCleaner(new MuonTrajectoryCleaner()),
  theTrackLoader(trackLoader) {
}

// Constructor, with user-defined cleaner. For the STA reconstruction the trackLoader must have the propagator.
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder,
				 MuonTrackLoader *trackLoader,
				 MuonTrajectoryCleaner* cleaner) :
  theTrajBuilder(ConcreteMuonTrajectoryBuilder),
  theTrajCleaner(cleaner),
  theTrackLoader(trackLoader) {
}

// destructor
MuonTrackFinder::~MuonTrackFinder() {

  LogTrace("Muon|RecoMuon|MuonTrackFinder")<<"MuonTrackFinder destructor called"<<endl;
  delete theTrajBuilder;
  delete theTrajCleaner;
  delete theTrackLoader;

}

// percolate the event setup
void MuonTrackFinder::setEvent(const Event& event) {
  theTrajBuilder->setEvent(event);
}

// convert the trajectories into tracks and load them in to the event
edm::OrphanHandle<reco::TrackCollection>  
MuonTrackFinder::load(const TrajectoryContainer& trajectories, 
		      edm::Event& event) {
  
  return theTrackLoader->loadTracks(trajectories, event);
}

// convert the trajectories into tracks and load them in to the event
void MuonTrackFinder::load(const CandidateContainer& muonCands,
			   Event& event) {
                           
    theTrackLoader->loadTracks(muonCands, event);

}

// reconstruct trajectories
edm::OrphanHandle<reco::TrackCollection>
MuonTrackFinder::reconstruct(const edm::Handle<edm::View<TrajectorySeed> >& seeds,
			     edm::Event& event){
  
  const string metname = "Muon|RecoMuon|MuonTrackFinder";
  LogTrace(metname)<<"SA Recostruction starting from: "<<seeds->size()<<endl;  
  
  // Percolate the event 
  setEvent(event);
  
  // Trajectory container
  TrajectoryContainer muonTrajectories;
  TrajectorySeedCollection::size_type nSeed = 0;
  // reconstruct the trajectory
  edm::View<TrajectorySeed>::const_iterator seed;
  for(seed = seeds->begin();
      seed != seeds->end(); ++seed, ++nSeed){
    LogTrace(metname)<<"+++ New Seed +++"<<endl;
    TrajectoryContainer muonTrajs_temp = theTrajBuilder->trajectories(*seed);
    for(TrajectoryContainer::iterator it = muonTrajs_temp.begin(); 
	it != muonTrajs_temp.end(); ++it){
      (*it)->setSeedRef(seeds->refAt(nSeed));
      muonTrajectories.push_back(*it); 
    }
  }
  
  // clean the clone traj
  LogTrace(metname)<<"Clean the trajectories container"<<endl;
  if(theTrajCleaner) theTrajCleaner->clean(muonTrajectories, event); //used by reference...
  
  // convert the trajectories into tracks and load them in to the event
  LogTrace(metname)
    <<"Convert the trajectories into tracks and load them in to the event"<<endl;
  return load(muonTrajectories,event);
  
}


// reconstruct trajectories
void MuonTrackFinder::reconstruct(const std::vector<TrackCand>& staCandColl,
				  Event& event){

  const string metname = "Muon|RecoMuon|MuonTrackFinder";

  // percolate the event 
  setEvent(event);

  // Muon Candidate container
  CandidateContainer muonCandidates;

  // reconstruct the muon candidates
  for (vector<TrackCand>::const_iterator staCand = staCandColl.begin(); staCand != staCandColl.end(); ++staCand) {
    CandidateContainer muonCands_temp = theTrajBuilder->trajectories(*staCand);
    muonCandidates.insert(muonCandidates.end(), muonCands_temp.begin(),muonCands_temp.end());
  }                                  
  
  // clean the cloned candidates
  if(theTrajCleaner) theTrajCleaner->clean(muonCandidates);

  // convert the trajectories into staTracks and load them into the event
  LogTrace(metname)<<"Load Muon Candidates into the event"<<endl;
  load(muonCandidates,event);

}
