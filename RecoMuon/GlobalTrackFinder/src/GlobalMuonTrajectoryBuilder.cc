/**
 *  Class: GlobalMuonTrajectoryBuilder
 *
 *  Description:
 *   Reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *
 *  $Date: 2007/08/17 15:31:36 $
 *  $Revision: 1.108 $
 *
 *  Authors :
 *  N. Neumeister            Purdue University
 *  C. Liu                   Purdue University
 *  A. Everett               Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *
 **/

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : GlobalTrajectoryBuilderBase(par, service) {

  theTkTrajsAvailableFlag = par.getParameter<bool>("TkTrajectoryAvailable");
  theFirstEvent = true;
  theTkTrackLabel = par.getParameter<edm::InputTag>("TrackerCollectionLabel");

}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {
}

//
// get information from event
//
void GlobalMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|setEvent";
  
  GlobalTrajectoryBuilderBase::setEvent(event);

  // get tracker TrackCollection from Event
  edm::Handle<std::vector<Trajectory> > handleTrackerTrajs;
  event.getByLabel(theTkTrackLabel,allTrackerTracks);
  LogInfo(category) 
      << "Found " << allTrackerTracks->size() 
      << " tracker Tracks with label "<< theTkTrackLabel;  
  if ( theTkTrajsAvailableFlag ) {
    event.getByLabel(theTkTrackLabel,handleTrackerTrajs);
    allTrackerTrajs = &*handleTrackerTrajs;  
   
    event.getByLabel(theTkTrackLabel,tkAssoMap);   
    if ( theFirstEvent ) {
      LogInfo(category) << "Tk Trajectories Found! ";
      theFirstEvent = false;
    }
  }

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|trajectories";
  
  // cut on muons with low momenta
  if ( (staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut || (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();
  
  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);
  addTraj(staCand);

  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogInfo(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";  

  // match tracker tracks to muon track
  vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  LogInfo(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if ( trackerTracks.empty() ) return CandidateContainer();

  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  LogInfo(category) << "turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    if ((*tkt).first != 0 && (*tkt).first->isValid()) {
      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, new Trajectory(*(*tkt).first));
      tkTrajs.push_back(muonCand);
      LogTrace(category) << "tpush";
    }
  }

  if ( tkTrajs.empty() )  {
    LogTrace(category) << "tkTrajs empty";
    return CandidateContainer();
  }

  CandidateContainer result = build(staCand, tkTrajs);
  LogInfo(category) << "Found "<< result.size() << " GLBMuons from one STACand";

  // free memory
  if ( staCandIn.first == 0) delete staCand.first;

  for( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); ++it) {
    if ( (*it)->trajectory() ) delete (*it)->trajectory();
    if ( (*it)->trackerTrajectory() ) delete (*it)->trackerTrajectory();
    if ( *it ) delete (*it);
  }
  tkTrajs.clear();  

  if ( !theTkTrajsAvailableFlag ) {
    for ( vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
      delete (*is).first;   
    }
  }

  return result;
  
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<GlobalMuonTrajectoryBuilder::TrackCand> GlobalMuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) const {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;
  
  vector<TrackCand> tkTrackCands;
  
  if ( theTkTrajsAvailableFlag ) {
    for(TrajTrackAssociationCollection::const_iterator it = tkAssoMap->begin(); it != tkAssoMap->end(); ++it){	
      const Ref<vector<Trajectory> > traj = it->key;
      const reco::TrackRef tk = it->val;
      TrackCand tkCand = TrackCand(0,tk);
      if( traj->isValid() ) tkCand.first = &*traj ;
      tkTrackCands.push_back(tkCand);
    }
  } else {
    for ( unsigned int position = 0; position != allTrackerTracks->size(); ++position ) {
      reco::TrackRef tkTrackRef(allTrackerTracks,position);
      TrackCand tkCand = TrackCand(0,tkTrackRef);
      tkTrackCands.push_back(tkCand); 
    }
  }
  
  tkCandColl = chooseRegionalTrackerTracks(staCand,tkTrackCands);
  
  return tkCandColl;
  
}
