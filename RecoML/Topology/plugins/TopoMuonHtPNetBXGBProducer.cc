/** \class TopoMuonHtPNetBXGBProducer
 *
 *  This class is an EDProducer that produces a single float value corresponding
 * to the output score of an XGBoost model of a "topological trigger" (TOPO) for
 * events with at least one muon + HT and b-tag. The model takes as input the
 * PFHT, the leading N PNetB scores among jets in the event (configurable), and
 * the pt and isolation variables of up to N muons (configurable, 0 = no muon
 * features).
 *
 *  \author Artur Lobanov – University of Hamburg
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/XGBoost/interface/XGBooster.h"

class TopoMuonHtPNetBXGBProducer : public edm::global::EDProducer<> {
public:
  using RecoChargedCandMap =
      edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoChargedCandidate>, float, unsigned int>>;

  // 1 global feature (PFHT) + nPNetB_ PNetB scores + 4 features per muon (pt,
  // tkIso, ecalIso, hcalIso)
  static constexpr unsigned int kFeaturesPerMuon = 4;

  explicit TopoMuonHtPNetBXGBProducer(edm::ParameterSet const&);
  ~TopoMuonHtPNetBXGBProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  /* Tokens */
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> chargedCandidatesToken_;
  const edm::EDGetTokenT<RecoChargedCandMap> ecalIsoMapToken_;
  const edm::EDGetTokenT<RecoChargedCandMap> hcalIsoMapToken_;
  const edm::EDGetTokenT<edm::ValueMap<double>> trackIsoMapToken_;
  const edm::EDGetTokenT<reco::METCollection> pfhtToken_;
  const edm::EDGetTokenT<reco::JetTagCollection> pnetToken_;

  /* Cuts */
  const double muonPtCut_;
  const double muonEtaCut_;

  /* Config */
  const unsigned int nMuons_;      // number of muons used as input features (0 = no muon features)
  const unsigned int nPNetB_;      // number of leading PNetB scores used as input features
  const unsigned int nFeatures_;   // 1 + nPNetB_ + kFeaturesPerMuon * nMuons_
  const bool muonSortByTkIso_;     // if true: ascending relative tkiso; if false:
                                   // descending pt
  const unsigned int nTreeLimit_;  // max number of trees to use in prediction
                                   // (0 = use all trees)

  /* XGBoost */
  std::unique_ptr<pat::XGBooster> booster_;
  const edm::EDPutTokenT<float> scoreToken_;

  const bool debug_;
};

using namespace edm;

TopoMuonHtPNetBXGBProducer::TopoMuonHtPNetBXGBProducer(edm::ParameterSet const& iConfig)
    : chargedCandidatesToken_(consumes(iConfig.getParameter<edm::InputTag>("ChargedCandidates"))),
      ecalIsoMapToken_(consumes(iConfig.getParameter<edm::InputTag>("EcalPFClusterIsoMap"))),
      hcalIsoMapToken_(consumes(iConfig.getParameter<edm::InputTag>("HcalPFClusterIsoMap"))),
      trackIsoMapToken_(consumes(iConfig.getParameter<edm::InputTag>("TrackIsoMap"))),
      pfhtToken_(consumes(iConfig.getParameter<edm::InputTag>("PFHT"))),
      pnetToken_(consumes(iConfig.getParameter<edm::InputTag>("PNetBscore"))),
      muonPtCut_(iConfig.getParameter<double>("muonPtCut")),
      muonEtaCut_(iConfig.getParameter<double>("muonEtaCut")),
      nMuons_(iConfig.getParameter<unsigned int>("nMuons")),
      nPNetB_(iConfig.getParameter<unsigned int>("nPNetB")),
      nFeatures_(1 + nPNetB_ + kFeaturesPerMuon * nMuons_),
      muonSortByTkIso_(iConfig.getParameter<bool>("muonSortByTkIso")),
      nTreeLimit_(iConfig.getParameter<unsigned int>("nTreeLimit")),
      scoreToken_(produces<float>("score")),
      debug_(iConfig.getParameter<bool>("debug")) {
  /* Load model */
  const edm::FileInPath modelPath(iConfig.getParameter<std::string>("modelPath"));

  if (debug_) {
    std::cout << "Loading XGBoost model from " << modelPath.fullPath() << "\n"
              << " nMuons=" << nMuons_ << " nPNetB=" << nPNetB_ << " nFeatures=" << nFeatures_
              << " muonSortByTkIso=" << muonSortByTkIso_ << " nTreeLimit=" << nTreeLimit_ << std::endl;
  }

  booster_ = std::make_unique<pat::XGBooster>(modelPath.fullPath());

  // Feature layout: [PFHT | PNetB_0 .. PNetB_nPNetB-1 | mu0_pt, mu0_tkIso,
  // mu0_ecalIso, mu0_hcalIso | mu1_... ]
  booster_->addFeature("pfht");
  for (unsigned int ib = 0; ib < nPNetB_; ++ib)
    booster_->addFeature("pnetb" + std::to_string(ib));
  for (unsigned int imu = 0; imu < nMuons_; ++imu) {
    booster_->addFeature("muon" + std::to_string(imu) + "_pt");
    booster_->addFeature("muon" + std::to_string(imu) + "_tkIso");
    booster_->addFeature("muon" + std::to_string(imu) + "_ecalIso");
    booster_->addFeature("muon" + std::to_string(imu) + "_hcalIso");
  }
}

/* ------------------------------------------------------------ */

void TopoMuonHtPNetBXGBProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& setup) const {
  float outScore = -1.f;
  // buffer for features to be fed to XGBoost; zero-padding for missing/empty
  // features
  std::vector<float> features(nFeatures_, 0.f);

  /* ---------------- PFHT ---------------- */

  float pfht = 0.f;

  if (const auto& h = iEvent.getHandle(pfhtToken_); h.isValid()) {
    if (!h->empty())
      pfht = h->front().sumEt();
  } else {
    LogWarning("TopoMuonHtPNetBXGBProducer") << "Missing PFHT collection";
  }

  /* ---------------- PNetB scores: collect and sort descending ----------------
   */

  std::vector<float> pnetScores;

  if (const auto& h = iEvent.getHandle(pnetToken_); h.isValid()) {
    pnetScores.reserve(h->size());
    for (auto const& tag : *h)
      pnetScores.push_back(tag.second);
    std::sort(pnetScores.begin(), pnetScores.end(), std::greater<float>());
  } else {
    LogWarning("TopoMuonHtPNetBXGBProducer") << "Missing PNetB JetTags";
  }

  /* ---------------- Fill global features ---------------- */

  // Feature layout: [PFHT | PNetB_0 .. PNetB_nPNetB-1 | muon blocks ]
  features[0] = pfht;

  const unsigned int nPNetBFill = std::min(pnetScores.size(), static_cast<size_t>(nPNetB_));
  for (unsigned int ib = 0; ib < nPNetBFill; ++ib)
    features[1 + ib] = pnetScores[ib];
  // remaining PNetB slots stay 0 (zero-padded) if fewer jets are found

  /* ---------------- Muons (optional: nMuons_=0 skips this block) -----------
   */

  std::vector<size_t> muonIndices;

  if (nMuons_ > 0) {
    const auto& muonsH = iEvent.getHandle(chargedCandidatesToken_);

    if (!muonsH.isValid()) {
      LogError("TopoMuonHtPNetBXGBProducer") << "Missing ChargedCandidates";
      iEvent.emplace(scoreToken_, outScore);
      return;
    }

    // Fetch track iso map once — needed for sorting and feature filling
    const auto& trkH = iEvent.getHandle(trackIsoMapToken_);

    muonIndices.reserve(muonsH->size());

    for (size_t i = 0; i < muonsH->size(); ++i) {
      const auto& mu = (*muonsH)[i];
      if (mu.pt() < muonPtCut_)
        continue;
      if (std::abs(mu.eta()) > muonEtaCut_)
        continue;
      muonIndices.push_back(i);
    }

    if (muonIndices.empty()) {
      iEvent.emplace(scoreToken_, outScore);
      return;
    }

    /* ---------------- Sort ---------------- */

    if (muonSortByTkIso_) {
      // Ascending relative track iso — tightest isolation first.
      // Muons with missing track iso map are pushed to the back.
      std::sort(muonIndices.begin(), muonIndices.end(), [&](size_t a, size_t b) {
        const float isoA =
            trkH.isValid() ? static_cast<float>((*trkH)[edm::getRef(muonsH, a)]) : std::numeric_limits<float>::max();
        const float isoB =
            trkH.isValid() ? static_cast<float>((*trkH)[edm::getRef(muonsH, b)]) : std::numeric_limits<float>::max();
        const float ptA = (*muonsH)[a].pt();
        const float ptB = (*muonsH)[b].pt();
        // if iso is 0 for both then sort by descending pt for
        // deterministic order
        if (isoA == 0.f && isoB == 0.f)
          return ptA > ptB;
        return (isoA / ptA) < (isoB / ptB);
      });
    } else {
      // Descending pt
      std::sort(muonIndices.begin(), muonIndices.end(), [&](size_t a, size_t b) {
        return (*muonsH)[a].pt() > (*muonsH)[b].pt();
      });
    }

    /* ---------------- Isolations (ECAL/HCAL) ---------------- */

    const auto& ecalH = iEvent.getHandle(ecalIsoMapToken_);
    const auto& hcalH = iEvent.getHandle(hcalIsoMapToken_);

    /* ---------------- Fill muon features ---------------- */

    const unsigned int muonOffset = 1 + nPNetB_;
    const unsigned int nMuonFill = std::min(muonIndices.size(), static_cast<size_t>(nMuons_));

    for (unsigned int m = 0; m < nMuonFill; ++m) {
      const size_t idx = muonIndices[m];
      const auto& mu = (*muonsH)[idx];
      const float pt = mu.pt();
      const auto muRef = edm::getRef(muonsH, idx);

      const float ecalIso = ecalH.isValid() ? (*ecalH)[muRef] : 10.f;
      const float hcalIso = hcalH.isValid() ? (*hcalH)[muRef] : 10.f;
      const float trkIso = trkH.isValid() ? static_cast<float>((*trkH)[muRef]) : 10.f;

      const unsigned int base = muonOffset + m * kFeaturesPerMuon;
      features[base + 0] = pt;
      features[base + 1] = trkIso;
      features[base + 2] = ecalIso;
      features[base + 3] = hcalIso;
    }
    // remaining muon slots stay 0 (zero-padded) if fewer muons are found
  }

  /* ---------------- XGBoost inference ---------------- */

  // nTreeLimit_: 0 = use all trees in the model (recommended when model was
  // saved after pruning to best iteration); set to best_iteration+1 when
  // the full un-pruned model is saved and early stopping was used
  outScore = booster_->predict(features, nTreeLimit_);

  /* ---------------- Debug ---------------- */

  if (debug_) {
    std::ostringstream ss;
    ss << "TopoMuonHtPNetBXGBProducer:"
       << " nPNetB(found)=" << pnetScores.size() << " nMuons(found)=" << muonIndices.size() << "\n Features: ";
    for (float f : features)
      ss << f << " ";
    ss << " --> score=" << outScore;
    std::cout << ss.str() << std::endl;
  }

  iEvent.emplace(scoreToken_, outScore);
}

/* ------------------------------------------------------------ */

void TopoMuonHtPNetBXGBProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFHT", edm::InputTag("hltPFHTJet30"));
  desc.add<edm::InputTag>("PNetBscore", edm::InputTag("hltParticleNetDiscriminatorsJetTags", "BvsAll"));
  desc.add<edm::InputTag>("ChargedCandidates", edm::InputTag("hltIterL3MuonCandidates"));
  desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltMuonEcalMFPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltMuonHcalRegPFClusterIsoForMuons"));
  desc.add<edm::InputTag>("TrackIsoMap",
                          edm::InputTag("hltMuonTkRelIsolationCut0p3Map", "combinedRelativeIsoDeposits"));
  desc.add<std::string>("modelPath",
                        "HLTrigger/HLTfilters/data/"
                        "HLT_xgb_model_HH2b2W1L_1mu_HLTHT_sorttkisoMupt-absiso_PNetB.json");

  desc.add<unsigned int>("nMuons", 1)
      ->setComment(
          "number of muons used as input features; 0 = no muon features (b-jet "
          "+ HT only model)");
  desc.add<unsigned int>("nPNetB", 1)
      ->setComment(
          "number of leading PNetB scores used as input features, sorted "
          "descending");
  desc.add<double>("muonPtCut", 10.0);
  desc.add<double>("muonEtaCut", 2.4);
  desc.add<bool>("muonSortByTkIso", true)
      ->setComment(
          "false: sort by descending pt, true: sort by ascending relative "
          "tkiso");
  desc.add<unsigned int>("nTreeLimit", 0)
      ->setComment(
          "max number of trees used in prediction; 0 = use all trees in the "
          "model");
  desc.add<bool>("debug", false);

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopoMuonHtPNetBXGBProducer);
