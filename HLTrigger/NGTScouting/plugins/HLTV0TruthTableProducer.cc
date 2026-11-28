// -*- C++ -*-
//
// Package:    HLTrigger/NGTScouting
// Class:      HLTV0TruthTableProducer
//
/**\class HLTV0TruthTableProducer HLTV0TruthTableProducer.cc HLTrigger/NGTScouting/plugins/HLTV0TruthTableProducer.cc

 Description: diagnostic nanoaod::FlatTable for the HLT V0 reconstruction efficiency.
              One row per generated K0s -> pi+ pi- or Lambda -> p pi- (and charge conjugate) of the
              signal event. The decay is located through the pair of daughter TrackingParticles
              sharing the same production vertex, whose summed momentum matches the generated V0.
              For each daughter the associated track in the input track collection (via the
              SimToReco association) is stored, together with the index of the reconstructed V0
              built from the same pair of tracks, if any.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

namespace {
  // pdgIds of the (positive, negative) daughters for a given V0 pdgId
  std::pair<int, int> daughterPdgIds(int v0PdgId) {
    switch (v0PdgId) {
      case 310:
        return {211, -211};
      case 3122:
        return {2212, -211};
      case -3122:
        return {211, -2212};
      default:
        return {0, 0};
    }
  }

  // key of the pair of daughter tracks, order-independent
  std::pair<int, int> trackPairKey(int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); }
}  // namespace

class HLTV0TruthTableProducer : public edm::global::EDProducer<> {
public:
  explicit HLTV0TruthTableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // map (sorted pair of daughter track keys) -> index of the reconstructed V0
  std::map<std::pair<int, int>, int16_t> recoV0Map(
      const edm::Event&, const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection>&) const;

  const edm::EDGetTokenT<reco::GenParticleCollection> genToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  const edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> kshortToken_;
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> lambdaToken_;
  const double maxDeltaR_;
  const double maxRelDeltaPt_;
  const std::string name_;
  const std::string doc_;
};

HLTV0TruthTableProducer::HLTV0TruthTableProducer(const edm::ParameterSet& params)
    : genToken_(consumes(params.getParameter<edm::InputTag>("genParticles"))),
      tpToken_(consumes(params.getParameter<edm::InputTag>("trackingParticles"))),
      simToRecoToken_(consumes(params.getParameter<edm::InputTag>("simToReco"))),
      bsToken_(consumes(params.getParameter<edm::InputTag>("beamSpot"))),
      kshortToken_(consumes(params.getParameter<edm::InputTag>("kshorts"))),
      lambdaToken_(consumes(params.getParameter<edm::InputTag>("lambdas"))),
      maxDeltaR_(params.getParameter<double>("maxDeltaR")),
      maxRelDeltaPt_(params.getParameter<double>("maxRelDeltaPt")),
      name_(params.getParameter<std::string>("name")),
      doc_(params.getParameter<std::string>("doc")) {
  produces<nanoaod::FlatTable>();
}

std::map<std::pair<int, int>, int16_t> HLTV0TruthTableProducer::recoV0Map(
    const edm::Event& iEvent, const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection>& token) const {
  std::map<std::pair<int, int>, int16_t> result;
  const auto handle = iEvent.getHandle(token);
  if (!handle.isValid())
    return result;
  for (size_t i = 0; i < handle->size(); ++i) {
    const auto& v0 = (*handle)[i];
    std::array<int, 2> keys{{-1, -1}};
    for (size_t id = 0; id < std::min<size_t>(v0.numberOfDaughters(), 2); ++id) {
      if (const auto* rcc = dynamic_cast<const reco::RecoChargedCandidate*>(v0.daughter(id));
          rcc && rcc->track().isNonnull())
        keys[id] = static_cast<int>(rcc->track().key());
    }
    if (keys[0] >= 0 && keys[1] >= 0)
      result.emplace(trackPairKey(keys[0], keys[1]), static_cast<int16_t>(i));
  }
  return result;
}

void HLTV0TruthTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  static constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  const auto genHandle = iEvent.getHandle(genToken_);
  const auto tpHandle = iEvent.getHandle(tpToken_);
  const auto simToRecoHandle = iEvent.getHandle(simToRecoToken_);
  const auto bsHandle = iEvent.getHandle(bsToken_);
  const reco::BeamSpot* bs = bsHandle.isValid() ? bsHandle.product() : nullptr;

  // generated V0s of the signal event
  std::vector<const reco::GenParticle*> genV0s;
  if (genHandle.isValid()) {
    for (const auto& gp : *genHandle) {
      if (daughterPdgIds(gp.pdgId()).first != 0)
        genV0s.push_back(&gp);
    }
  } else {
    edm::LogWarning("HLTV0TruthTableProducer") << "Invalid handle for the GenParticle input collection";
  }
  const size_t nGen = genV0s.size();

  std::vector<float> v_pt(nGen, kNaN), v_eta(nGen, kNaN), v_phi(nGen, kNaN);
  std::vector<int> v_pdgId(nGen, 0);
  std::vector<uint8_t> v_decayFound(nGen, 0);
  std::vector<float> v_x(nGen, kNaN), v_y(nGen, kNaN), v_z(nGen, kNaN), v_lxy(nGen, kNaN);
  std::vector<float> v_posPt(nGen, kNaN), v_posEta(nGen, kNaN), v_posD0(nGen, kNaN);
  std::vector<float> v_negPt(nGen, kNaN), v_negEta(nGen, kNaN), v_negD0(nGen, kNaN);
  std::vector<int16_t> v_posTrkIdx(nGen, -1), v_negTrkIdx(nGen, -1), v_recoV0Idx(nGen, -1);
  std::vector<uint8_t> v_posNPixelHits(nGen, 0), v_negNPixelHits(nGen, 0);
  std::vector<uint8_t> v_posNStripHits(nGen, 0), v_negNStripHits(nGen, 0);

  // signal, in-time charged pions and protons, grouped by production vertex
  std::map<unsigned int, std::vector<size_t>> tpsByVertex;
  if (tpHandle.isValid()) {
    for (size_t i = 0; i < tpHandle->size(); ++i) {
      const auto& tp = (*tpHandle)[i];
      if (tp.eventId().event() != 0 || tp.eventId().bunchCrossing() != 0)
        continue;
      const int apdg = std::abs(tp.pdgId());
      if ((apdg != 211 && apdg != 2212) || tp.parentVertex().isNull())
        continue;
      tpsByVertex[tp.parentVertex().key()].push_back(i);
    }
  } else {
    edm::LogWarning("HLTV0TruthTableProducer") << "Invalid handle for the TrackingParticle input collection";
  }

  const auto kshortMap = recoV0Map(iEvent, kshortToken_);
  const auto lambdaMap = recoV0Map(iEvent, lambdaToken_);

  // transverse impact parameter of a TrackingParticle w.r.t. the beamline (straight-line approximation)
  auto tpD0 = [bs](const TrackingParticle& tp) -> float {
    const double bx = bs != nullptr ? bs->x(tp.vz()) : 0.;
    const double by = bs != nullptr ? bs->y(tp.vz()) : 0.;
    return tp.pt() > 0. ? std::abs((tp.vx() - bx) * tp.py() - (tp.vy() - by) * tp.px()) / tp.pt() : kNaN;
  };

  // associated track (highest association quality) for a TrackingParticle
  auto associatedTrack = [&](size_t tpIdx) -> const edm::RefToBase<reco::Track>* {
    if (!simToRecoHandle.isValid())
      return nullptr;
    const auto it = simToRecoHandle->find(TrackingParticleRef(tpHandle, tpIdx));
    if (it == simToRecoHandle->end() || it->val.empty())
      return nullptr;
    return &it->val.front().first;
  };

  for (size_t i = 0; i < nGen; ++i) {
    const auto& gen = *genV0s[i];
    v_pt[i] = gen.pt();
    v_eta[i] = gen.eta();
    v_phi[i] = gen.phi();
    v_pdgId[i] = gen.pdgId();

    // find the pair of daughter TrackingParticles best matching the generated V0
    const auto [posPdg, negPdg] = daughterPdgIds(gen.pdgId());
    double bestDR = maxDeltaR_;
    size_t bestPos = 0, bestNeg = 0;
    bool found = false;
    for (const auto& [vtxKey, tpIdxs] : tpsByVertex) {
      for (const size_t ip : tpIdxs) {
        if ((*tpHandle)[ip].pdgId() != posPdg)
          continue;
        for (const size_t in : tpIdxs) {
          if ((*tpHandle)[in].pdgId() != negPdg)
            continue;
          const auto sum = (*tpHandle)[ip].p4() + (*tpHandle)[in].p4();
          if (std::abs(sum.pt() - gen.pt()) > maxRelDeltaPt_ * gen.pt())
            continue;
          const double dR = reco::deltaR(sum.eta(), sum.phi(), gen.eta(), gen.phi());
          if (dR < bestDR) {
            bestDR = dR;
            bestPos = ip;
            bestNeg = in;
            found = true;
          }
        }
      }
    }
    if (!found)
      continue;

    const auto& posTP = (*tpHandle)[bestPos];
    const auto& negTP = (*tpHandle)[bestNeg];
    v_decayFound[i] = 1;
    v_x[i] = posTP.vx();
    v_y[i] = posTP.vy();
    v_z[i] = posTP.vz();
    const double bx = bs != nullptr ? bs->x(posTP.vz()) : 0.;
    const double by = bs != nullptr ? bs->y(posTP.vz()) : 0.;
    v_lxy[i] = std::hypot(posTP.vx() - bx, posTP.vy() - by);
    v_posPt[i] = posTP.pt();
    v_posEta[i] = posTP.eta();
    v_posD0[i] = tpD0(posTP);
    v_negPt[i] = negTP.pt();
    v_negEta[i] = negTP.eta();
    v_negD0[i] = tpD0(negTP);

    if (const auto* trk = associatedTrack(bestPos)) {
      v_posTrkIdx[i] = static_cast<int16_t>(trk->key());
      v_posNPixelHits[i] = static_cast<uint8_t>((*trk)->hitPattern().numberOfValidPixelHits());
      v_posNStripHits[i] = static_cast<uint8_t>((*trk)->hitPattern().numberOfValidStripHits());
    }
    if (const auto* trk = associatedTrack(bestNeg)) {
      v_negTrkIdx[i] = static_cast<int16_t>(trk->key());
      v_negNPixelHits[i] = static_cast<uint8_t>((*trk)->hitPattern().numberOfValidPixelHits());
      v_negNStripHits[i] = static_cast<uint8_t>((*trk)->hitPattern().numberOfValidStripHits());
    }

    if (v_posTrkIdx[i] >= 0 && v_negTrkIdx[i] >= 0) {
      const auto& recoMap = (std::abs(gen.pdgId()) == 310) ? kshortMap : lambdaMap;
      const auto it = recoMap.find(trackPairKey(v_posTrkIdx[i], v_negTrkIdx[i]));
      if (it != recoMap.end())
        v_recoV0Idx[i] = it->second;
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nGen, name_, /*singleton*/ false);
  table->setDoc(doc_);
  table->addColumn<float>("pt", v_pt, "generated V0 p_T (GeV)", 10);
  table->addColumn<float>("eta", v_eta, "generated V0 #eta", 12);
  table->addColumn<float>("phi", v_phi, "generated V0 #phi (rad)", 12);
  table->addColumn<int>("pdgId", v_pdgId, "PDG id (310: K0s, +/-3122: Lambda/anti-Lambda)");
  table->addColumn<uint8_t>(
      "decayFound", v_decayFound, "1 if the pair of daughter TrackingParticles matching the V0 was found");
  table->addColumn<float>("x", v_x, "decay vertex x position (cm)", 10);
  table->addColumn<float>("y", v_y, "decay vertex y position (cm)", 10);
  table->addColumn<float>("z", v_z, "decay vertex z position (cm)", 14);
  table->addColumn<float>("lxy", v_lxy, "transverse decay length w.r.t. the beamspot (cm)", 10);
  table->addColumn<float>("posPt", v_posPt, "positive daughter TrackingParticle p_T (GeV)", 10);
  table->addColumn<float>("posEta", v_posEta, "positive daughter TrackingParticle #eta", 12);
  table->addColumn<float>(
      "posD0",
      v_posD0,
      "positive daughter TrackingParticle |d0| w.r.t. the beamline, straight-line approximation (cm)",
      10);
  table->addColumn<int16_t>("posTrkIdx",
                            v_posTrkIdx,
                            "index of the track associated to the positive daughter in the input track collection "
                            "(-1 if not reconstructed)");
  table->addColumn<uint8_t>(
      "posNPixelHits", v_posNPixelHits, "number of valid pixel hits of the track associated to the positive daughter");
  table->addColumn<uint8_t>(
      "posNStripHits",
      v_posNStripHits,
      "number of valid strip (phase-2: Outer Tracker) hits of the track associated to the positive daughter");
  table->addColumn<float>("negPt", v_negPt, "negative daughter TrackingParticle p_T (GeV)", 10);
  table->addColumn<float>("negEta", v_negEta, "negative daughter TrackingParticle #eta", 12);
  table->addColumn<float>(
      "negD0",
      v_negD0,
      "negative daughter TrackingParticle |d0| w.r.t. the beamline, straight-line approximation (cm)",
      10);
  table->addColumn<int16_t>("negTrkIdx",
                            v_negTrkIdx,
                            "index of the track associated to the negative daughter in the input track collection "
                            "(-1 if not reconstructed)");
  table->addColumn<uint8_t>(
      "negNPixelHits", v_negNPixelHits, "number of valid pixel hits of the track associated to the negative daughter");
  table->addColumn<uint8_t>(
      "negNStripHits",
      v_negNStripHits,
      "number of valid strip (phase-2: Outer Tracker) hits of the track associated to the negative daughter");
  table->addColumn<int16_t>(
      "recoV0Idx",
      v_recoV0Idx,
      "index of the reconstructed V0 (hltKshort or hltLambda table) built from the two associated tracks (-1 if none)");

  iEvent.put(std::move(table));
}

void HLTV0TruthTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"))->setComment("generated particles");
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("TrackingParticles");
  desc.add<edm::InputTag>("simToReco", edm::InputTag("tpToHltGeneralTrackAssociation"))
      ->setComment("reco::SimToRecoCollection association between TrackingParticles and the input tracks");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"))->setComment("beamspot");
  desc.add<edm::InputTag>("kshorts", edm::InputTag("hltGeneralV0Candidates", "Kshort"))
      ->setComment("reconstructed K0s candidates");
  desc.add<edm::InputTag>("lambdas", edm::InputTag("hltGeneralV0Candidates", "Lambda"))
      ->setComment("reconstructed Lambda candidates");
  desc.add<double>("maxDeltaR", 0.05)
      ->setComment("maximum deltaR between the generated V0 and the sum of the daughter TrackingParticles");
  desc.add<double>("maxRelDeltaPt", 0.1)
      ->setComment("maximum relative p_T difference between the generated V0 and the sum of the daughters");
  desc.add<std::string>("name", "hltGenV0")->setComment("name of the flat table output");
  desc.add<std::string>("doc", "generated K0s and Lambda with their daughters and HLT reconstruction status")
      ->setComment("a few words of documentation");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HLTV0TruthTableProducer);
