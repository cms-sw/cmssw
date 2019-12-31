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
 *  Authors :
 *  N. Neumeister            Purdue University
 *  C. Liu                   Purdue University
 *  A. Everett               Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *
 **/

#include "RecoMuon/L3TrackFinder/interface/L3MuonTrajectoryBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

//----------------
// Constructors --
//----------------
L3MuonTrajectoryBuilder::L3MuonTrajectoryBuilder(const edm::ParameterSet& par,
                                                 const MuonServiceProxy* service,
                                                 edm::ConsumesCollector& iC)
    : GlobalTrajectoryBuilderBase(par, service, iC) {
  theTrajectoryCleaner = std::make_unique<TrajectoryCleanerBySharedHits>();
  theTkCollName = par.getParameter<edm::InputTag>("tkTrajLabel");
  theBeamSpotInputTag = par.getParameter<edm::InputTag>("tkTrajBeamSpot");
  theMaxChi2 = par.getParameter<double>("tkTrajMaxChi2");
  theDXYBeamSpot = par.getParameter<double>("tkTrajMaxDXYBeamSpot");
  theUseVertex = par.getParameter<bool>("tkTrajUseVertex");
  theVertexCollInputTag = par.getParameter<edm::InputTag>("tkTrajVertex");
  theTrackToken = iC.consumes<reco::TrackCollection>(theTkCollName);
}

//--------------
// Destructor --
//--------------
L3MuonTrajectoryBuilder::~L3MuonTrajectoryBuilder() {}

void L3MuonTrajectoryBuilder::fillDescriptions(edm::ParameterSetDescription& desc) {
  edm::ParameterSetDescription descTRB;
  MuonTrackingRegionBuilder::fillDescriptionsHLT(descTRB);
  desc.add("MuonTrackingRegionBuilder", descTRB);
}

//
// Get information from event
//
void L3MuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|setEvent";

  GlobalTrajectoryBuilderBase::setEvent(event);

  // get tracker TrackCollection from Event
  event.getByToken(theTrackToken, allTrackerTracks);
  LogDebug(category) << "Found " << allTrackerTracks->size() << " tracker Tracks with label " << theTkCollName;

  if (theUseVertex) {
    // PV
    edm::Handle<reco::VertexCollection> pvHandle;
    if (pvHandle.isValid()) {
      vtx = pvHandle->front();
    } else {
      edm::LogInfo(category) << "No Primary Vertex available from EventSetup \n";
    }
  } else {
    // BS
    event.getByLabel(theBeamSpotInputTag, beamSpotHandle);
    if (beamSpotHandle.isValid()) {
      beamSpot = *beamSpotHandle;
    } else {
      edm::LogInfo(category) << "No beam spot available from EventSetup \n";
    }
  }
}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer L3MuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {
  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|trajectories";

  // cut on muons with low momenta
  if ((staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut ||
      (staCandIn).second->innerMomentum().R() < 2.5)
    return CandidateContainer();

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);

  std::vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogDebug(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";

  // match tracker tracks to muon track
  std::vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);

  LogDebug(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if (trackerTracks.empty())
    return CandidateContainer();

  // build a combined tracker-muon MuonCandidate
  // turn tkMatchedTracks into MuonCandidates
  LogDebug(category) << "turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  tkTrajs.reserve(trackerTracks.size());
  for (std::vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    if ((*tkt).first != nullptr && (*tkt).first->isValid()) {
      tkTrajs.emplace_back(std::make_unique<MuonCandidate>(
          nullptr, staCand.second, (*tkt).second, std::make_unique<Trajectory>(*(*tkt).first)));
    } else {
      tkTrajs.emplace_back(std::make_unique<MuonCandidate>(nullptr, staCand.second, (*tkt).second, nullptr));
    }
  }

  if (tkTrajs.empty()) {
    LogDebug(category) << "tkTrajs empty";
    return CandidateContainer();
  }

  CandidateContainer result = build(staCand, tkTrajs);
  LogDebug(category) << "Found " << result.size() << " L3Muons from one L2Cand";

  // free memory
  if (staCandIn.first == nullptr)
    delete staCand.first;

  for (std::vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
    delete (*is).first;
  }

  return result;
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
std::vector<L3MuonTrajectoryBuilder::TrackCand> L3MuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) {
  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|makeTkCandCollection";
  std::vector<TrackCand> tkCandColl;
  std::vector<TrackCand> tkTrackCands;

  //  for (auto&& tkTrack: allTrackerTracks){
  //    auto tkCand = TrackCand((Trajectory*)(0),tkTrack);
  for (unsigned int position = 0; position != allTrackerTracks->size(); ++position) {
    reco::TrackRef tkTrackRef(allTrackerTracks, position);
    TrackCand tkCand = TrackCand((Trajectory*)nullptr, tkTrackRef);
    tkCandColl.push_back(tkCand);
  }

  //Loop over TrackCand collection made from allTrackerTracks in previous step
  for (auto&& tkCand : tkCandColl) {
    auto& tk = tkCand.second;
    bool canUseL3MTS = false;
    // check the seedRef is non-null first; and then
    if (tk->seedRef().isNonnull()) {
      auto a = dynamic_cast<const L3MuonTrajectorySeed*>(tk->seedRef().get());
      canUseL3MTS = a != nullptr;
    }
    if (canUseL3MTS) {
      edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef =
          tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
      // May still need provenance here, so using trackref:
      reco::TrackRef staTrack = l3seedRef->l2Track();
      if (staTrack == (staCand.second)) {
        // Apply filters (dxy, chi2 cut)
        double tk_vtx;
        if (theUseVertex)
          tk_vtx = tk->dxy(vtx.position());
        else
          tk_vtx = tk->dxy(beamSpot.position());
        if (fabs(tk_vtx) > theDXYBeamSpot || tk->normalizedChi2() > theMaxChi2)
          continue;
        tkTrackCands.push_back(tkCand);
      }
    } else {
      // We will try to match all tracker tracks with the muon:
      double tk_vtx;
      if (theUseVertex)
        tk_vtx = tk->dxy(vtx.position());
      else
        tk_vtx = tk->dxy(beamSpot.position());
      if (fabs(tk_vtx) > theDXYBeamSpot || tk->normalizedChi2() > theMaxChi2)
        continue;
      tkTrackCands.push_back(tkCand);
    }
  }

  return tkTrackCands;
}
