/**
 *  Class: L3MuonTrajectoryBuilder
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
 *  $Date: 2009/07/29 12:21:11 $
 *  $Revision: 1.16 $
 *
 *  Authors :
 *  N. Neumeister            Purdue University
 *  C. Liu                   Purdue University
 *  A. Everett               Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *
 **/

#include "RecoMuon/L3TrackFinder/interface/L3MuonTrajectoryBuilder.h"


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
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

L3MuonTrajectoryBuilder::L3MuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : GlobalTrajectoryBuilderBase(par, service) {

  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    

  theTkCollName = par.getParameter<edm::InputTag>("tkTrajLabel");

}


//--------------
// Destructor --
//--------------

L3MuonTrajectoryBuilder::~L3MuonTrajectoryBuilder() {
  if (theTrajectoryCleaner) delete theTrajectoryCleaner;
}

//
// get information from event
//
void L3MuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  
  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|setEvent";
  
  GlobalTrajectoryBuilderBase::setEvent(event);
      
  // get tracker TrackCollection from Event
  event.getByLabel(theTkCollName,allTrackerTracks);
  LogDebug(category) 
      << "Found " << allTrackerTracks->size() 
      << " tracker Tracks with label "<< theTkCollName;  
  
}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer L3MuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|trajectories";
  
  // cut on muons with low momenta
  if ( (staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut || (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();
  
  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);
  
  vector<TrackCand> trackerTracks;
  
  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogDebug(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";  
  
  // match tracker tracks to muon track
  trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  
  LogDebug(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if ( trackerTracks.empty() ) return CandidateContainer();
  
  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  LogDebug(category) << "turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    if ((*tkt).first != 0 && (*tkt).first->isValid()) {
      //
      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, new Trajectory(*(*tkt).first));
      tkTrajs.push_back(muonCand);
      //      LogTrace(category) << "tpush";
      //
    } else {
      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, 0);
      tkTrajs.push_back(muonCand);
    }
  }
    
  if ( tkTrajs.empty() )  {
    LogDebug(category) << "tkTrajs empty";
    return CandidateContainer();
  }
  
  CandidateContainer result = build(staCand, tkTrajs);  
  LogDebug(category) << "Found "<< result.size() << " L3Muons from one L2Cand";

  // free memory
  if ( staCandIn.first == 0) delete staCand.first;

  for( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); ++it) {
    if ( (*it)->trajectory() ) delete (*it)->trajectory();
    if ( (*it)->trackerTrajectory() ) delete (*it)->trackerTrajectory();
    if ( *it ) delete (*it);
  }
  tkTrajs.clear();  

  for ( vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
    delete (*is).first;   
  }
  
  return result;
  
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<L3MuonTrajectoryBuilder::TrackCand> L3MuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;
  
  vector<TrackCand> tkTrackCands;
  
  for ( unsigned int position = 0; position != allTrackerTracks->size(); ++position ) {
    reco::TrackRef tkTrackRef(allTrackerTracks,position);
    TrackCand tkCand = TrackCand((Trajectory*)(0),tkTrackRef);
    tkCandColl.push_back(tkCand);
  }

  for(vector<TrackCand>::const_iterator tk = tkCandColl.begin(); tk != tkCandColl.end() ; ++tk) { 
    edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = (*tk).second->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >() ;
    reco::TrackRef staTrack = l3seedRef->l2Track();
    if(staTrack == (staCand.second) ) tkTrackCands.push_back(*tk);
  }

  return tkTrackCands;
  
}

