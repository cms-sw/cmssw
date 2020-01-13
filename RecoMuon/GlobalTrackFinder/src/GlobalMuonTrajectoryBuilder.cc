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

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par,
                                                         const MuonServiceProxy* service,
                                                         edm::ConsumesCollector& iC)
    : GlobalTrajectoryBuilderBase(par, service, iC)

{
  theTkTrackLabel = par.getParameter<edm::InputTag>("TrackerCollectionLabel");
  allTrackerTracksToken = iC.consumes<reco::TrackCollection>(theTkTrackLabel);
}

//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {}

//
// get information from event
//
void GlobalMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|setEvent";

  GlobalTrajectoryBuilderBase::setEvent(event);

  // get tracker TrackCollection from Event
  event.getByToken(allTrackerTracksToken, allTrackerTracks);
  LogDebug(category) << " Found " << allTrackerTracks->size() << " tracker Tracks with label " << theTkTrackLabel;
}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|trajectories";

  // cut on muons with low momenta
  LogTrace(category) << " STA pt " << staCandIn.second->pt() << " rho " << staCandIn.second->innerMomentum().Rho()
                     << " R " << staCandIn.second->innerMomentum().R() << " theCut " << thePtCut;

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);

  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogTrace(category) << " Found " << regionalTkTracks.size() << " tracks within region of interest";

  // match tracker tracks to muon track
  vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  LogTrace(category) << " Found " << trackerTracks.size() << " matching tracker tracks within region of interest";

  if (trackerTracks.empty()) {
    if (staCandIn.first == nullptr)
      delete staCand.first;

    return CandidateContainer();
  }

  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  LogTrace(category) << " Turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    tkTrajs.push_back(std::make_unique<MuonCandidate>(nullptr, staCand.second, (*tkt).second, nullptr));
  }

  if (tkTrajs.empty()) {
    LogTrace(category) << " tkTrajs empty";
    if (staCandIn.first == nullptr)
      delete staCand.first;

    return CandidateContainer();
  }

  CandidateContainer result = build(staCand, tkTrajs);
  LogTrace(category) << " Found " << result.size() << " GLBMuons from one STACand";

  // free memory
  if (staCandIn.first == nullptr)
    delete staCand.first;

  return result;
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<GlobalMuonTrajectoryBuilder::TrackCand> GlobalMuonTrajectoryBuilder::makeTkCandCollection(
    const TrackCand& staCand) {
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;

  vector<TrackCand> tkTrackCands;

  for (unsigned int position = 0; position != allTrackerTracks->size(); ++position) {
    reco::TrackRef tkTrackRef(allTrackerTracks, position);
    TrackCand tkCand = TrackCand((Trajectory*)nullptr, tkTrackRef);
    tkTrackCands.push_back(tkCand);
  }

  tkCandColl = chooseRegionalTrackerTracks(staCand, tkTrackCands);

  return tkCandColl;
}
