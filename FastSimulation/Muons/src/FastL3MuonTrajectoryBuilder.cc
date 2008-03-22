/**
 *  Class: FastL3MuonTrajectoryBuilder
 *
 *  Description:
 *   Reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information) and from already reconstructed tracker tracks.
 *   It performs matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *
 *  $Date: 2008/02/14 12:34:38 $
 *  $Revision: 1.4 $
 *
 *  Authors :
 *  Patrick Janot - CERN
 *
 **/

#include "FastSimulation/Muons/interface/FastL3MuonTrajectoryBuilder.h"


//---------------
// C++ Headers --
//---------------

#include <iostream>
//#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// Tracker RecHits and Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 

// SimTrack
#include "SimDataFormats/Track/interface/SimTrack.h"

//----------------
// Constructors --
//----------------

FastL3MuonTrajectoryBuilder::FastL3MuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) 
: 
  GlobalTrajectoryBuilderBase(par, service), 
  theTkSeedGenerator(0), theTrajectoryCleaner(0) 
{

  theTkTrajsAvailableFlag = par.getParameter<bool>("TkTrajectoryAvailable");  
  theFirstEvent = true;
  
  theTrackerTrajectoryCollection = par.getParameter<edm::InputTag>("TrackerTrajectories");
  theSimModule = par.getParameter<edm::InputTag>("SimulatedMuons");

  //
  // start seed generator;
  //
  std::string stateOnTrackerOutProp = 
    par.getParameter<std::string>("StateOnTrackerBoundOutPropagator");

}


//--------------
// Destructor --
//--------------

FastL3MuonTrajectoryBuilder::~FastL3MuonTrajectoryBuilder() {
  if (theTrajectoryCleaner) delete theTrajectoryCleaner;
}

//
// get information from event
//
void FastL3MuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  
  // const std::string category = "Muon|RecoMuon|FastL3MuonTrajectoryBuilder|setEvent";
    
  GlobalTrajectoryBuilderBase::setEvent(event);
  theEvent = &event;
    
  // Retrieve tracker tracks for muons
  regionalTkTracks = makeTkCandCollection(dummyStaCand);
  // edm::LogInfo(category) << "Found " << regionalTkTracks.size() << " tracker tracks from Muons";  

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer 
FastL3MuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {

  // static const std::string category = "FastL3MuonTrajectoryBuilder";

  // cut on muons with low momenta
  if ( (staCandIn).second->pt() < thePtCut || 
       (staCandIn).second->innerMomentum().Rho() < thePtCut || 
       (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);
  addTraj(staCand);
  
  // match tracker tracks to muon track
  std::vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  // edm::LogInfo(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if ( trackerTracks.empty() ) return CandidateContainer();
  
  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  //edm::LogInfo(category) << "turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (std::vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    if ((*tkt).first != 0 && (*tkt).first->isValid()) {
      std::vector<Trajectory> refittedTk;
      Trajectory refittedTkTraj = *(*tkt).first;
      refittedTk = refitTrajectory(*(*tkt).first);
      if(refittedTk.size() == 1) refittedTkTraj = refittedTk.front();
      
      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, new Trajectory(refittedTkTraj));
      tkTrajs.push_back(muonCand);
      // LogTrace(category) << "tpush";

    }
  }
  
  if ( tkTrajs.empty() )  {
    // edm::LogInfo(category) << "tkTrajs empty";
    return CandidateContainer();
  }
  
  CandidateContainer result = build(staCand, tkTrajs);  
  // edm::LogInfo(category) << "Found "<< result.size() << " L3Muons from one L2Cand";

  // free memory
  if ( staCandIn.first == 0) delete staCand.first;

  for( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); ++it) {
    if ( (*it)->trajectory() ) delete (*it)->trajectory();
    if ( (*it)->trackerTrajectory() ) delete (*it)->trackerTrajectory();
    if ( *it ) delete (*it);
  }
  tkTrajs.clear();  

  return result;
  
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
std::vector<FastL3MuonTrajectoryBuilder::TrackCand> 
FastL3MuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) const {

  // const std::string category = "Muon|RecoMuon|FastL3MuonTrajectoryBuilder|makeTkCandCollection";

  std::vector<TrackCand> tkCandColl;  

  // Get the tracker tracks corresponding to simulated muons 
  edm:: Handle<std::vector<Trajectory> > theTrajectories;
  edm::Handle<TrajTrackAssociationCollection> theAssoMap;  
  edm::Handle<std::vector<SimTrack> > simMuons;
  theEvent->getByLabel(theTrackerTrajectoryCollection,theTrajectories);
  theEvent->getByLabel(theTrackerTrajectoryCollection,theAssoMap);
  theEvent->getByLabel(theSimModule,simMuons);
  unsigned nmuons = simMuons->size();
  
  TrajTrackAssociationCollection::const_iterator anAssociation;  
  TrajTrackAssociationCollection::const_iterator firstAssociation;  
  TrajTrackAssociationCollection::const_iterator lastAssociation;
  firstAssociation = theAssoMap->begin();
  lastAssociation = theAssoMap->end();

  // Loop on simulated muons
  for ( unsigned amuon=0; amuon<nmuons; ++amuon ) {
    const SimTrack& simTrack = (*simMuons)[amuon];

    for ( anAssociation = firstAssociation; anAssociation != lastAssociation; ++anAssociation ) { 
      edm::Ref<std::vector<Trajectory> > aTrajectoryRef = anAssociation->key;
      reco::TrackRef aTrackRef = anAssociation->val;
      int recoTrackId = findId(*aTrackRef);
      if ( recoTrackId == (int)(simTrack.trackId()) ) {
	tkCandColl.push_back(TrackCand(new Trajectory((*aTrajectoryRef)),reco::TrackRef()));
	break;
      }
    }
  }
	
  // LogTrace(category) << "Found " << tkCandColl.size() << " tkCands from seeds";

  return tkCandColl;

}


//
// build a tracker Trajectory from a seed
//
std::vector<Trajectory> 
FastL3MuonTrajectoryBuilder::makeTrajsFromSeeds(const std::vector<TrajectorySeed>& tkSeeds) const {

  // const std::string category = "Muon|RecoMuon|FastL3MuonTrajectoryBuilder|makeTrajsFromSeeds";
  std::vector<Trajectory> result;
  // edm::LogInfo(category) << "Trajectories from all seeds " << result.size();
  return result;

}

int 
FastL3MuonTrajectoryBuilder::findId(const reco::Track& aTrack) const {
  int trackId = -1;
  trackingRecHit_iterator aHit = aTrack.recHitsBegin();
  trackingRecHit_iterator lastHit = aTrack.recHitsEnd();
  for ( ; aHit!=lastHit; ++aHit ) {
    if ( !aHit->get()->isValid() ) continue;
    const SiTrackerGSRecHit2D * rechit = (const SiTrackerGSRecHit2D*) (aHit->get());
    trackId = rechit->simtrackId();
    break;
  }
  return trackId;
}

void 
FastL3MuonTrajectoryBuilder::clear() { 
  std::vector<TrackCand>::const_iterator is = regionalTkTracks.begin();
  std::vector<TrackCand>::const_iterator il = regionalTkTracks.end();
  for ( ; is != il; ++is) {
    delete (*is).first;   
  }
}




