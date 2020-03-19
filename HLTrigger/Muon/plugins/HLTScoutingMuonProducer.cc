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
      TrackCollection_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      EcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>("EcalPFClusterIsoMap"))),
      HcalPFClusterIsoMap_(consumes<RecoChargedCandMap>(iConfig.getParameter<edm::InputTag>("HcalPFClusterIsoMap"))),
      TrackIsoMap_(consumes<edm::ValueMap<double>>(iConfig.getParameter<edm::InputTag>("TrackIsoMap"))),
      displacedvertexCollection_(
          consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("displacedvertexCollection"))),
      muonPtCut(iConfig.getParameter<double>("muonPtCut")),
      muonEtaCut(iConfig.getParameter<double>("muonEtaCut")),
      minVtxProbCut(iConfig.getParameter<double>("minVtxProbCut")),
      linkToken_(consumes<reco::MuonTrackLinksCollection>(iConfig.getParameter<edm::InputTag>("InputLinks"))) {
  //register products
  produces<ScoutingMuonCollection>();
  produces<ScoutingVertexCollection>("displacedVtx");
}

HLTScoutingMuonProducer::~HLTScoutingMuonProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingMuonProducer::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const {
  using namespace edm;

  std::unique_ptr<ScoutingMuonCollection> outMuons(new ScoutingMuonCollection());
  std::unique_ptr<ScoutingVertexCollection> dispVertices(new ScoutingVertexCollection());

  // Get RecoChargedCandidate
  Handle<reco::RecoChargedCandidateCollection> ChargedCandidateCollection;
  if (!iEvent.getByToken(ChargedCandidateCollection_, ChargedCandidateCollection)) {
    iEvent.put(std::move(outMuons));
    return;
  }

  // Get Track
  Handle<reco::TrackCollection> TrackCollection;
  if (!iEvent.getByToken(TrackCollection_, TrackCollection)) {
    iEvent.put(std::move(outMuons));
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
  if (!iEvent.getByToken(TrackIsoMap_, TrackIsoMap)) {
    iEvent.put(std::move(outMuons));
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

  // Produce muons
  std::vector<int> vtxInd;
  float minDR2 = 1e-06;
  int index = 0;

  // Read Links collection:
  edm::Handle<reco::MuonTrackLinksCollection> links;
  iEvent.getByToken(linkToken_, links);

  for (auto& muon : *ChargedCandidateCollection) {
    reco::RecoChargedCandidateRef muonRef = getRef(ChargedCandidateCollection, index);
    ++index;
    if (muonRef.isNull() || !muonRef.isAvailable())
      continue;

    reco::TrackRef track = muon.track();
    if (track.isNull() || !track.isAvailable())
      continue;

    int validmuhit = 0;
    int matchedsta = 0;
    for (auto const& link : *links) {
      const reco::Track& trackerTrack = *link.trackerTrack();
      float dR2 = deltaR2(track->eta(), track->phi(), trackerTrack.eta(), trackerTrack.phi());
      float dPt = std::abs(track->pt() - trackerTrack.pt());
      if (track->pt() != 0)
        dPt = dPt / track->pt();

      if (dR2 < 0.02 * 0.02 and dPt < 0.001) {
        const reco::TrackRef staTrack = link.standAloneTrack();
        validmuhit = staTrack->hitPattern().numberOfValidMuonHits();
        matchedsta = staTrack->hitPattern().muonStationsWithValidHits();
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
                           ecalisopf,
                           hcalisopf,
                           (*TrackIsoMap)[muonRef],
                           track->chi2(),
                           track->ndof(),
                           track->charge(),
                           track->dxy(),
                           track->dz(),
                           validmuhit,
                           track->hitPattern().numberOfValidPixelHits(),
                           matchedsta,
                           track->hitPattern().trackerLayersWithMeasurement(),
                           2,  // Global muon
                           track->hitPattern().numberOfValidStripHits(),
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
  desc.add<edm::InputTag>("ChargedCandidates", edm::InputTag("hltIterL3MuonCandidates"));
  desc.add<edm::InputTag>("Tracks", edm::InputTag("hltPixelTracks"));
  desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltMuonEcalMFPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltMuonHcalRegPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("TrackIsoMap", edm::InputTag("hltMuonTkRelIsolationCut0p07Map:combinedRelativeIsoDeposits"));
  desc.add<edm::InputTag>("displacedvertexCollection", edm::InputTag("hltPixelVertices"));
  desc.add<double>("muonPtCut", 3.0);
  desc.add<double>("muonEtaCut", 2.4);
  desc.add<double>("minVtxProbCut", 0.001);
  desc.add<edm::InputTag>("InputLinks", edm::InputTag("hltL3MuonsIterL3Links"));

  descriptions.add("hltScoutingMuonProducer", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingMuonProducer);
