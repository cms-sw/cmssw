/** \class HLTTopoMuonHtPNetBXGBProducer
 *
 *  This class is an EDProducer that produces a single float value corresponding to the output score of an XGBoost model
 *  of a "topological trigger" (TOPO) for events with at least one muon + HT and b-tag. 
 *  The model takes as input the PFHT, 
 *  the maximum PNetB score among jets in the event, 
 *  and the pt and isolation variables of up to N muons (configurable).
 *
 *  \author Artur Lobanov – University of Hamburg
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/getRef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this class header
#include "HLTTopoMuonHtPNetBXGBProducer.hh"

using namespace edm;

HLTTopoMuonHtPNetBXGBProducer::HLTTopoMuonHtPNetBXGBProducer(
    edm::ParameterSet const& iConfig)

    : chargedCandidatesToken_(
          consumes(iConfig.getParameter<edm::InputTag>("ChargedCandidates"))),
      ecalIsoMapToken_(
          consumes(iConfig.getParameter<edm::InputTag>("EcalPFClusterIsoMap"))),
      hcalIsoMapToken_(
          consumes(iConfig.getParameter<edm::InputTag>("HcalPFClusterIsoMap"))),
      trackIsoMapToken_(
          consumes(iConfig.getParameter<edm::InputTag>("TrackIsoMap"))),
      pfhtToken_(consumes(iConfig.getParameter<edm::InputTag>("PFHT"))),
      pnetToken_(consumes(iConfig.getParameter<edm::InputTag>("PNetBscore"))),
      muonPtCut_(iConfig.getParameter<double>("muonPtCut")),
      muonEtaCut_(iConfig.getParameter<double>("muonEtaCut")),
      nMuons_(iConfig.getParameter<unsigned int>("nMuons")),
      nFeatures_(kGlobalFeatures + kFeaturesPerMuon * nMuons_),
      muonSortByTkIso_(iConfig.getParameter<bool>("muonSortByTkIso")),
      buffer_(nFeatures_, 0.f),
      debug_(iConfig.getParameter<bool>("debug")) {
  produces<float>("score");

  /* Load model */
  const edm::FileInPath modelPath(
      iConfig.getParameter<std::string>("modelPath"));

  if (debug_) {
    LogInfo("HLTTopoMuonHtPNetBXGBProducer")
    // std::cout 
              << "Loading XGBoost model from " << modelPath.fullPath()
              << std::endl
              << " nMuons=" << nMuons_ << " nFeatures=" << nFeatures_
              << " muonSortByTkIso=" << muonSortByTkIso_ << std::endl;
  }

  XGBoosterCreate(nullptr, 0, &booster_);
  XGBoosterLoadModel(booster_, modelPath.fullPath().c_str());

  XGDMatrixCreateFromMat(buffer_.data(), 1, nFeatures_, -999.f, &dmat_);

  xgbConfig_ =
      "{\"training\": false, \"type\": 0, "
      "\"iteration_begin\": 0, \"iteration_end\": 0, "
      "\"strict_shape\": false}";
}

/* ------------------------------------------------------------ */

HLTTopoMuonHtPNetBXGBProducer::~HLTTopoMuonHtPNetBXGBProducer() {
  if (dmat_) XGDMatrixFree(dmat_);
  if (booster_) XGBoosterFree(booster_);
}

/* ------------------------------------------------------------ */

void HLTTopoMuonHtPNetBXGBProducer::produce(edm::Event& iEvent,
                                            edm::EventSetup const&) {
  float outScore = -1.f;

  /* ---------------- Muons: collect passing cuts ---------------- */

  const auto muonsH = iEvent.getHandle(chargedCandidatesToken_);

  if (!muonsH.isValid()) {
    LogError("HLTTopoMuonHtPNetBXGBProducer")
        << "Missing ChargedCandidates";
    iEvent.put(std::make_unique<float>(outScore), "score");
    return;
  }

  // Fetch track iso map once — needed for sorting and feature filling
  const auto trkH = iEvent.getHandle(trackIsoMapToken_);

  std::vector<size_t> muonIndices;
  muonIndices.reserve(muonsH->size());

  for (size_t i = 0; i < muonsH->size(); ++i) {
    const auto& mu = (*muonsH)[i];
    if (mu.pt() < muonPtCut_) continue;
    if (std::abs(mu.eta()) > muonEtaCut_) continue;
    muonIndices.push_back(i);
  }

  if (muonIndices.empty()) {
    iEvent.put(std::make_unique<float>(outScore), "score");
    return;
  }

  /* ---------------- Sort ---------------- */

  if (muonSortByTkIso_) {
    // Ascending track iso — tightest isolation first.
    // Muons with missing track iso map are pushed to the back.
    std::sort(muonIndices.begin(), muonIndices.end(), [&](size_t a, size_t b) {
      const float isoA =
          trkH.isValid() ? static_cast<float>((*trkH)[edm::getRef(muonsH, a)])
                         : std::numeric_limits<float>::max();
      const float isoB =
          trkH.isValid() ? static_cast<float>((*trkH)[edm::getRef(muonsH, b)])
                         : std::numeric_limits<float>::max();
      // get pt of muons too to make relative isolation sorting
      const float ptA = (*muonsH)[a].pt();
      const float ptB = (*muonsH)[b].pt();

      // if iso is 0 for both then sort by descending pt to have a deterministic order, otherwise sort by ascending relative isolation
      if (isoA == 0.f && isoB == 0.f) return ptA > ptB;
      return (isoA / ptA) < (isoB / ptB);
    });
  } else {
    // Descending pt
    std::sort(muonIndices.begin(), muonIndices.end(), [&](size_t a, size_t b) {
      return (*muonsH)[a].pt() > (*muonsH)[b].pt();
    });
  }

  /* ---------------- Isolations (ECAL/HCAL) ---------------- */

  const auto ecalH = iEvent.getHandle(ecalIsoMapToken_);
  const auto hcalH = iEvent.getHandle(hcalIsoMapToken_);

  /* ---------------- PFHT ---------------- */

  float pfht = 0.f;

  if (auto h = iEvent.getHandle(pfhtToken_); h.isValid()) {
    if (!h->empty()) pfht = h->front().sumEt();
  } else {
    LogWarning("HLTTopoMuonHtPNetBXGBProducer")
        << "Missing PFHT collection";
  }

  /* ---------------- PNetB ---------------- */

  float maxPNetB = -1.f;

  if (auto h = iEvent.getHandle(pnetToken_); h.isValid()) {
    for (auto const& tag : *h) maxPNetB = std::max(maxPNetB, tag.second);
  } else {
    LogWarning("HLTTopoMuonHtPNetBXGBProducer") << "Missing PNetB JetTags";
  }

  /* ---------------- Fill buffer ---------------- */

  // Reset to zero so padding slots for missing muons are well-defined
  std::fill(buffer_.begin(), buffer_.end(), 0.f);

  buffer_[0] = pfht;
  buffer_[1] = maxPNetB;

  const unsigned int nFill =
      std::min(muonIndices.size(), static_cast<size_t>(nMuons_));

  for (unsigned int m = 0; m < nFill; ++m) {
    const size_t idx = muonIndices[m];
    const auto& mu = (*muonsH)[idx];
    const float pt = mu.pt();
    const auto muRef = edm::getRef(muonsH, idx);

    const float ecalIso = ecalH.isValid() ? (*ecalH)[muRef] : 10.f;
    const float hcalIso = hcalH.isValid() ? (*hcalH)[muRef] : 10.f;
    const float trkIso =
        trkH.isValid() ? static_cast<float>((*trkH)[muRef]) : 10.f;

    const unsigned int base = kGlobalFeatures + m * kFeaturesPerMuon;
    buffer_[base + 0] = pt;
    buffer_[base + 1] = trkIso;
    buffer_[base + 2] = ecalIso;
    buffer_[base + 3] = hcalIso;
  }

  /* ---------------- XGBoost inference ---------------- */

  XGDMatrixFree(dmat_);
  XGDMatrixCreateFromMat(buffer_.data(), 1, nFeatures_, -999.f, &dmat_);

  uint64_t const* outShape = nullptr;
  uint64_t outDim = 0;
  const float* outResult = nullptr;

  XGBoosterPredictFromDMatrix(booster_, dmat_, xgbConfig_.c_str(), &outShape,
                              &outDim, &outResult);

  if (outResult != nullptr) outScore = outResult[0];

  /* ---------------- Debug ---------------- */

  if (debug_) {
    std::ostringstream ss;
    ss << "HLTTopoMuonHtPNetBXGBProducer: nMuons(found)=" << muonIndices.size()
       << std::endl;
    ss << " Features: ";
    for (float f : buffer_) ss << f << " ";
    ss << " --> score=" << outScore;
    LogInfo("HLTTopoMuonHtPNetBXGBProducer")
    // std::cout 
      << ss.str() << std::endl;
  }

  iEvent.put(std::make_unique<float>(outScore), "score");
}

/* ------------------------------------------------------------ */

void HLTTopoMuonHtPNetBXGBProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFHT", edm::InputTag("hltPFHTJet30"));
  desc.add<edm::InputTag>(
      "PNetBscore",
      edm::InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"));
  desc.add<edm::InputTag>("ChargedCandidates",
                          edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("EcalPFClusterIsoMap",
                          edm::InputTag("hltMuonEcalPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("HcalPFClusterIsoMap",
                          edm::InputTag("hltMuonHcalPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("TrackIsoMap",
                          edm::InputTag("hltMuonTkRelIsolationCut0p09Map",
                                        "combinedRelativeIsoDeposits"));
  desc.add<std::string>(
      "modelPath",
      "L1Trigger/TopoML/data/HLT_xgb_model_HH2b2W1L_1mu_HLTHT_Mu_pt-iso_PNetB.json");

  desc.add<unsigned int>("nMuons", 1);
  desc.add<double>("muonPtCut", 10.0);
  desc.add<double>("muonEtaCut", 2.4);
  desc.add<bool>("muonSortByTkIso", false);  // false: sort by descending pt
                                             // true:  sort by ascending tkiso
  desc.add<bool>("debug", false);

  descriptions.add("HLTTopoMuonHtPNetBXGBProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTopoMuonHtPNetBXGBProducer);