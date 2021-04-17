// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingMuonProducer
//
/**\class HLTScoutingMuonProducer HLTScoutingMuonProducer.cc HLTScoutingMuonProducer.cc

Description: Producer for ScoutingMuon

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Fri, 31 Jul 2015
//
//

#include "HLTScoutingMuonProducer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TMath.h"

//
// constructors and destructor
//
HLTScoutingMuonProducer::HLTScoutingMuonProducer(const edm::ParameterSet& iConfig)
    : ChargedCandidateCollection_(
          consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("ChargedCandidates"))),
      displacedvertexCollection_(
          consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("displacedvertexCollection"))),
      MuonCollection_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("InputMuons"))),
      linkToken_(consumes<reco::MuonTrackLinksCollection>(iConfig.getParameter<edm::InputTag>("InputLinks"))),
      TrackCollection_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      EcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>("EcalPFClusterIsoMap"))),
      HcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>("HcalPFClusterIsoMap"))),
      TrackIsoMap_(consumes<edm::ValueMap<double>>(iConfig.getParameter<edm::InputTag>("TrackIsoMap"))),
      muonPtCut(iConfig.getParameter<double>("muonPtCut")),
      muonEtaCut(iConfig.getParameter<double>("muonEtaCut")),
      minVtxProbCut(iConfig.getParameter<double>("minVtxProbCut")) {
  //register products
  produces<Run3ScoutingMuonCollection>();
  produces<Run3ScoutingVertexCollection>("displacedVtx");
}

HLTScoutingMuonProducer::~HLTScoutingMuonProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingMuonProducer::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const {
  using namespace edm;

  std::unique_ptr<Run3ScoutingMuonCollection> outMuons(new Run3ScoutingMuonCollection());
  std::unique_ptr<Run3ScoutingVertexCollection> dispVertices(new Run3ScoutingVertexCollection());

  // Get RecoChargedCandidate
  Handle<reco::RecoChargedCandidateCollection> ChargedCandidateCollection;
  if (!iEvent.getByToken(ChargedCandidateCollection_, ChargedCandidateCollection)) {
    iEvent.put(std::move(outMuons));
    iEvent.put(std::move(dispVertices), "displacedVtx");
    return;
  }

  std::pair<reco::RecoChargedCandidate, reco::RecoChargedCandidate> ivtxMuPair;
  std::vector<std::pair<reco::RecoChargedCandidate, reco::RecoChargedCandidate>> vtxMuPair;

  //get displaced vertices
  Handle<reco::VertexCollection> displacedvertexCollection;
  if (iEvent.getByToken(displacedvertexCollection_, displacedvertexCollection)) {
    for (auto& dispvtx : *displacedvertexCollection) {
      if (!dispvtx.isValid())
        continue;
      float vtxProb = 0.0;
      if ((dispvtx.chi2() >= 0.0) && (dispvtx.ndof() > 0))
        vtxProb = TMath::Prob(dispvtx.chi2(), dispvtx.ndof());
      if (vtxProb < minVtxProbCut)
        continue;

      // Get the 2 tracks associated to displaced vertex
      auto trackIt = dispvtx.tracks_begin();
      reco::TrackRef vertextkRef1 = (*trackIt).castTo<reco::TrackRef>();
      trackIt++;
      reco::TrackRef vertextkRef2 = (*trackIt).castTo<reco::TrackRef>();

      // Get the muons associated with the tracks
      int iFoundRefs = 0;
      for (auto const& cand : *ChargedCandidateCollection) {
        reco::TrackRef tkRef = cand.get<reco::TrackRef>();
        if (tkRef == vertextkRef1) {
          ivtxMuPair.first = cand;
          iFoundRefs++;
        }
        if (tkRef == vertextkRef2) {
          ivtxMuPair.second = cand;
          iFoundRefs++;
        }
      }
      if (iFoundRefs < 2)
        continue;
      vtxMuPair.push_back(ivtxMuPair);

      dispVertices->emplace_back(dispvtx.x(),
                                 dispvtx.y(),
                                 dispvtx.z(),
                                 dispvtx.zError(),
                                 dispvtx.xError(),
                                 dispvtx.yError(),
                                 dispvtx.tracksSize(),
                                 dispvtx.chi2(),
                                 dispvtx.ndof(),
                                 dispvtx.isValid());
    }
  }

  // Get Muon collection
  Handle<reco::MuonCollection> MuonCollection;
  if (!iEvent.getByToken(MuonCollection_, MuonCollection)) {
    iEvent.put(std::move(outMuons));
    iEvent.put(std::move(dispVertices), "displacedVtx");
    return;
  }

  // Get Links collection
  edm::Handle<reco::MuonTrackLinksCollection> links;
  if (!iEvent.getByToken(linkToken_, links)) {
    iEvent.put(std::move(outMuons));
    iEvent.put(std::move(dispVertices), "displacedVtx");
    return;
  }

  // Get EcalPFClusterIsoMap
  Handle<RecoChargedCandMap> EcalPFClusterIsoMap;
  iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap);

  // Get HcalPFClusterIsoMap
  Handle<RecoChargedCandMap> HcalPFClusterIsoMap;
  iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap);

  // Get TrackIsoMap
  Handle<ValueMap<double>> TrackIsoMap;
  iEvent.getByToken(TrackIsoMap_, TrackIsoMap);

  // Produce muons
  std::vector<int> vtxInd;
  float minDR2 = 1e-06;
  int index = 0;

  for (auto& muon : *ChargedCandidateCollection) {
    reco::RecoChargedCandidateRef muonRef = getRef(ChargedCandidateCollection, index);
    ++index;
    if (muonRef.isNull() || !muonRef.isAvailable())
      continue;

    reco::TrackRef track = muon.track();
    if (track.isNull() || !track.isAvailable())
      continue;

    int validStandAloneMuonHits = 0;
    int matchedStandAloneMuonStations = 0;
    for (auto const& link : *links) {
      const reco::Track& trackerTrack = *link.trackerTrack();
      float dR2 = deltaR2(track->eta(), track->phi(), trackerTrack.eta(), trackerTrack.phi());
      float dPt = std::abs(track->pt() - trackerTrack.pt());
      if (track->pt() != 0)
        dPt = dPt / track->pt();

      if (dR2 < 0.02 * 0.02 and dPt < 0.001) {
        if (link.standAloneTrack().isNonnull()) {
          validStandAloneMuonHits = link.standAloneTrack()->hitPattern().numberOfValidMuonHits();
          matchedStandAloneMuonStations = link.standAloneTrack()->hitPattern().muonStationsWithValidHits();
        }
      }
    }

    unsigned int recoMuonType = 2;  // Global muon
    float normalizedChi2 = 999.0;
    int nRecoMuonValidMuonHits = 0;
    int nRecoMuonChambers = 0;
    int nRecoMuonChambersCSCorDT = 0;
    int nRecoMuonMatches = 0;
    int nRecoMuonMatchedStations = 0;
    unsigned int nRecoMuonExpectedMatchedStations = 0;
    unsigned int recoMuonStationMask = 0;
    int nRecoMuonMatchedRPCLayers = 0;
    unsigned int recoMuonRPClayerMask = 0;
    for (auto const& recoMu : *MuonCollection) {
      float dR2 = deltaR2(muon.eta(), muon.phi(), recoMu.eta(), recoMu.phi());
      float dPt = std::abs(muon.pt() - recoMu.pt());
      if (muon.pt() != 0)
        dPt = dPt / muon.pt();

      if (dR2 < 0.02 * 0.02 and dPt < 0.001) {
        if (recoMu.globalTrack().isNonnull()) {
          normalizedChi2 = recoMu.globalTrack()->normalizedChi2();
          nRecoMuonValidMuonHits = recoMu.globalTrack()->hitPattern().numberOfValidMuonHits();
        }
        recoMuonType = recoMu.type();
        nRecoMuonChambers = recoMu.numberOfChambers();
        nRecoMuonChambersCSCorDT = recoMu.numberOfChambersCSCorDT();
        nRecoMuonMatches = recoMu.numberOfMatches();
        nRecoMuonMatchedStations = recoMu.numberOfMatchedStations();
        nRecoMuonExpectedMatchedStations = recoMu.expectedNnumberOfMatchedStations();
        recoMuonStationMask = recoMu.stationMask();
        nRecoMuonMatchedRPCLayers = recoMu.numberOfMatchedRPCLayers();
        recoMuonRPClayerMask = recoMu.RPClayerMask();
      }
    }

    if (muon.pt() < muonPtCut)
      continue;
    if (fabs(muon.eta()) > muonEtaCut)
      continue;

    double ecalisopf = -1.0;
    if (EcalPFClusterIsoMap.isValid()) {
      ecalisopf = (*EcalPFClusterIsoMap)[muonRef];
    }

    double hcalisopf = -1.0;
    if (HcalPFClusterIsoMap.isValid()) {
      hcalisopf = (*HcalPFClusterIsoMap)[muonRef];
    }

    double trackiso = -1.0;
    if (TrackIsoMap.isValid()) {
      trackiso = (*TrackIsoMap)[muonRef];
    }

    vtxInd.reserve(vtxMuPair.size());
    for (unsigned int i = 0; i < vtxMuPair.size(); i++) {
      float dr2_1 = reco::deltaR2(((vtxMuPair[i]).first), muon);
      float dr2_2 = reco::deltaR2(((vtxMuPair[i]).second), muon);
      if ((dr2_1 < minDR2) || (dr2_2 < minDR2))
        vtxInd.push_back(i);
    }

    outMuons->emplace_back(muon.pt(),
                           muon.eta(),
                           muon.phi(),
                           muon.mass(),
                           recoMuonType,
                           track->charge(),
                           normalizedChi2,
                           ecalisopf,
                           hcalisopf,
                           trackiso,
                           validStandAloneMuonHits,
                           matchedStandAloneMuonStations,
                           nRecoMuonValidMuonHits,
                           nRecoMuonChambers,
                           nRecoMuonChambersCSCorDT,
                           nRecoMuonMatches,
                           nRecoMuonMatchedStations,
                           nRecoMuonExpectedMatchedStations,
                           recoMuonStationMask,
                           nRecoMuonMatchedRPCLayers,
                           recoMuonRPClayerMask,
                           track->hitPattern().numberOfValidPixelHits(),
                           track->hitPattern().numberOfValidStripHits(),
                           track->hitPattern().pixelLayersWithMeasurement(),
                           track->hitPattern().trackerLayersWithMeasurement(),
                           track->chi2(),
                           track->ndof(),
                           track->dxy(),
                           track->dz(),
                           track->qoverp(),
                           track->lambda(),
                           track->pt(),
                           track->phi(),
                           track->eta(),
                           track->dxyError(),
                           track->dzError(),
                           track->qoverpError(),
                           track->lambdaError(),
                           track->phiError(),
                           track->dsz(),
                           track->dszError(),
                           track->covariance(0, 1),
                           track->covariance(0, 2),
                           track->covariance(0, 3),
                           track->covariance(0, 4),
                           track->covariance(1, 2),
                           track->covariance(1, 3),
                           track->covariance(1, 4),
                           track->covariance(2, 3),
                           track->covariance(2, 4),
                           track->covariance(3, 4),
                           track->vx(),
                           track->vy(),
                           track->vz(),
                           track->hitPattern(),
                           vtxInd);
    vtxInd.clear();
  }

  // Put output
  iEvent.put(std::move(outMuons));
  iEvent.put(std::move(dispVertices), "displacedVtx");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ChargedCandidates", edm::InputTag("hltIterL3MuonCandidatesNoVtx"));
  desc.add<edm::InputTag>("displacedvertexCollection", edm::InputTag("hltDisplacedmumuVtxProducer"));
  desc.add<edm::InputTag>("InputMuons", edm::InputTag("hltIterL3MuonsNoVtx"));
  desc.add<edm::InputTag>("InputLinks", edm::InputTag("hltL3MuonsIterL3LinksNoVtx"));
  desc.add<edm::InputTag>("Tracks", edm::InputTag("hltPixelTracks"));
  desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltMuonEcalMFPFClusterIsoForMuonsNoVtx"));
  desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltMuonHcalPFClusterIsoForMuonsNoVtx"));
  desc.add<edm::InputTag>("TrackIsoMap",
                          edm::InputTag("hltMuonTkRelIsolationCut0p09MapNoVtx:combinedRelativeIsoDeposits"));
  desc.add<double>("muonPtCut", 3.0);
  desc.add<double>("muonEtaCut", 2.4);
  desc.add<double>("minVtxProbCut", 0.001);

  descriptions.add("hltScoutingMuonProducer", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingMuonProducer);
