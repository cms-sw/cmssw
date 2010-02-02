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
 *  $Date: 2009/07/29 12:16:18 $
 *  $Revision: 1.116 $
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
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1.h>
#include <TFile.h>

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : GlobalTrajectoryBuilderBase(par, service)
	   
{

  theTkTrackLabel = par.getParameter<edm::InputTag>("TrackerCollectionLabel");

  useTFileService_ = par.getUntrackedParameter<bool>("UseTFileService",false);

  if(useTFileService_) {
    edm::Service<TFileService> fs;
    TFileDirectory subDir = fs->mkdir( "builder" );
    h_nRegionalTk = subDir.make<TH1F>("h_nRegionalTk","Regional Tracker Tracks per STA",21,-0.5,20.5);
    h_nMatchedTk = subDir.make<TH1F>("h_nMatchedTk","Matched Tracker Tracks per STA",21,-0.5,20.5);
    h_nSta = subDir.make<TH1F>("h_nSta","Cut STA Muons",21,-0.5,20.5);
    h_nGlb = subDir.make<TH1F>("h_nGlb","Number of GLB per STA",21,-0.5,20.5);
    h_staPt = subDir.make<TH1F>("h_staPt","STA p_{T}",200,0,100);
    h_staRho = subDir.make<TH1F>("h_staRho","STA #rho",200,0,100);
    h_staR = subDir.make<TH1F>("h_staR","STA R",200,0,100);
  } else {
    h_nRegionalTk = 0;
    h_nMatchedTk = 0;
    h_nSta = 0;
    h_staPt = 0;
    h_staRho = 0;
    h_staR = 0;
    h_nGlb = 0;
  }
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
  event.getByLabel(theTkTrackLabel,allTrackerTracks);
  LogDebug(category) 
      << " Found " << allTrackerTracks->size() 
      << " tracker Tracks with label "<< theTkTrackLabel;  

}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|trajectories";

  // cut on muons with low momenta
  LogTrace(category) << " STA pt " << staCandIn.second->pt() << " rho " << staCandIn.second->innerMomentum().Rho() << " R " << staCandIn.second->innerMomentum().R() << " theCut " << thePtCut;

  if(h_nSta) h_nSta->Fill(1); 
  if(h_staPt) h_staPt->Fill((staCandIn).second->pt());
  if(h_staRho) h_staRho->Fill((staCandIn).second->innerMomentum().Rho());
  if(h_staR) h_staR->Fill((staCandIn).second->innerMomentum().R());

  if ( (staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut || (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();

  if(h_nSta) h_nSta->Fill(2); 

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);

  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogTrace(category) << " Found " << regionalTkTracks.size() << " tracks within region of interest";
  if(h_nRegionalTk) h_nRegionalTk->Fill(regionalTkTracks.size());

  // match tracker tracks to muon track
  vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  LogTrace(category) << " Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if(h_nMatchedTk) h_nMatchedTk->Fill(trackerTracks.size());

  if ( trackerTracks.empty() ) {
    if ( staCandIn.first == 0) delete staCand.first;

    return CandidateContainer();
  }

  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  LogTrace(category) << " Turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {

      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, 0);
      tkTrajs.push_back(muonCand);
  }

  if ( tkTrajs.empty() )  {
    LogTrace(category) << " tkTrajs empty";
    if ( staCandIn.first == 0) delete staCand.first;

    return CandidateContainer();
  }

  CandidateContainer result = build(staCand, tkTrajs);
  LogTrace(category) << " Found "<< result.size() << " GLBMuons from one STACand";

  if(h_nGlb) h_nGlb->Fill(result.size());

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
vector<GlobalMuonTrajectoryBuilder::TrackCand> GlobalMuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;
  
  vector<TrackCand> tkTrackCands;
    
  for ( unsigned int position = 0; position != allTrackerTracks->size(); ++position ) {
    reco::TrackRef tkTrackRef(allTrackerTracks,position);
    TrackCand tkCand = TrackCand((Trajectory*)(0),tkTrackRef);
    tkTrackCands.push_back(tkCand); 
  }
  
  
  tkCandColl = chooseRegionalTrackerTracks(staCand,tkTrackCands);
  
  return tkCandColl;
  
}
