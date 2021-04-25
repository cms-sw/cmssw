#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <cmath>  // std::fabs

L1HPSPFTauProducer::L1HPSPFTauProducer(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      tauBuilder_(nullptr),
      useChargedPFCandSeeds_(cfg.getParameter<bool>("useChargedPFCandSeeds")),
      minSeedChargedPFCandPt_(cfg.getParameter<double>("minSeedChargedPFCandPt")),
      maxSeedChargedPFCandEta_(cfg.getParameter<double>("maxSeedChargedPFCandEta")),
      maxSeedChargedPFCandDz_(cfg.getParameter<double>("maxSeedChargedPFCandDz")),
      useJetSeeds_(cfg.getParameter<bool>("useJetSeeds")),
      minSeedJetPt_(cfg.getParameter<double>("minSeedJetPt")),
      maxSeedJetEta_(cfg.getParameter<double>("maxSeedJetEta")),
      minPFTauPt_(cfg.getParameter<double>("minPFTauPt")),
      maxPFTauEta_(cfg.getParameter<double>("maxPFTauEta")),
      minLeadChargedPFCandPt_(cfg.getParameter<double>("minLeadChargedPFCandPt")),
      maxLeadChargedPFCandEta_(cfg.getParameter<double>("maxLeadChargedPFCandEta")),
      maxLeadChargedPFCandDz_(cfg.getParameter<double>("maxLeadChargedPFCandDz")),
      maxChargedIso_(cfg.getParameter<double>("maxChargedIso")),
      maxChargedRelIso_(cfg.getParameter<double>("maxChargedRelIso")),
      deltaRCleaning_(cfg.getParameter<double>("deltaRCleaning")),
      applyPreselection_(cfg.getParameter<bool>("applyPreselection")),
      debug_(cfg.getUntrackedParameter<bool>("debug", false)) {
  if (debug_) {
    std::cout << "<L1HPSPFTauProducer::L1HPSPFTauProducer (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;
  }

  tauBuilder_.reset(new L1HPSPFTauBuilder(cfg));

  srcL1PFCands_ = cfg.getParameter<edm::InputTag>("srcL1PFCands");
  tokenL1PFCands_ = consumes<l1t::PFCandidateCollection>(srcL1PFCands_);
  srcL1Jets_ = cfg.getParameter<edm::InputTag>("srcL1Jets");
  if (useJetSeeds_) {
    tokenL1Jets_ = consumes<std::vector<reco::CaloJet>>(srcL1Jets_);
  }
  srcL1Vertices_ = cfg.getParameter<edm::InputTag>("srcL1Vertices");
  if (srcL1Vertices_.label() != "") {
    tokenL1Vertices_ = consumes<std::vector<l1t::TkPrimaryVertex>>(srcL1Vertices_);
  }
  deltaR2Cleaning_ = deltaRCleaning_ * deltaRCleaning_;

  edm::ParameterSet cfg_signalQualityCuts = cfg.getParameter<edm::ParameterSet>("signalQualityCuts");
  signalQualityCutsDzCutDisabled_ = readL1PFTauQualityCuts(cfg_signalQualityCuts, "disabled");
  edm::ParameterSet cfg_isolationQualityCuts = cfg.getParameter<edm::ParameterSet>("isolationQualityCuts");
  isolationQualityCutsDzCutDisabled_ = readL1PFTauQualityCuts(cfg_isolationQualityCuts, "disabled");

  produces<l1t::L1HPSPFTauCollection>();
}

L1HPSPFTauProducer::~L1HPSPFTauProducer() {}

namespace {
  bool isHigherPt_pfCandRef(const l1t::PFCandidateRef& l1PFCand1, const l1t::PFCandidateRef& l1PFCand2) {
    return l1PFCand1->pt() > l1PFCand2->pt();
  }

  bool isHigherPt_pfTau(const l1t::L1HPSPFTau& l1PFTau1, const l1t::L1HPSPFTau& l1PFTau2) {
    return l1PFTau1.pt() > l1PFTau2.pt();
  }
}  // namespace

void L1HPSPFTauProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::unique_ptr<l1t::L1HPSPFTauCollection> l1PFTauCollectionCleaned(new l1t::L1HPSPFTauCollection());

  edm::Handle<l1t::PFCandidateCollection> l1PFCands;
  evt.getByToken(tokenL1PFCands_, l1PFCands);

  l1t::TkPrimaryVertexRef primaryVertex;
  float primaryVertex_z = 0.;
  if (srcL1Vertices_.label() != "") {
    edm::Handle<std::vector<l1t::TkPrimaryVertex>> vertices;
    evt.getByToken(tokenL1Vertices_, vertices);
    if (vertices->size() > 0) {
      primaryVertex = l1t::TkPrimaryVertexRef(vertices, 0);
      primaryVertex_z = primaryVertex->zvertex();
    }
  }

  if (debug_) {
    std::cout << "BEFORE selection:" << std::endl;
    for (auto l1PFCand : *l1PFCands) {
      printPFCand(std::cout, l1PFCand, primaryVertex_z);
    }
  }

  // build collection of selected PFCandidates
  std::vector<l1t::PFCandidateRef> selectedL1PFCandsSignalQualityCuts;
  std::vector<l1t::PFCandidateRef> selectedL1PFCandsSignalOrIsolationQualityCuts;
  size_t numL1PFCands = l1PFCands->size();
  for (size_t idxL1PFCand = 0; idxL1PFCand < numL1PFCands; ++idxL1PFCand) {
    l1t::PFCandidateRef l1PFCand(l1PFCands, idxL1PFCand);
    bool passesSignalQualityCuts = isSelected(signalQualityCutsDzCutDisabled_, *l1PFCand, primaryVertex_z);
    bool passesIsolationQualityCuts = isSelected(isolationQualityCutsDzCutDisabled_, *l1PFCand, primaryVertex_z);
    if (passesSignalQualityCuts) {
      selectedL1PFCandsSignalQualityCuts.push_back(l1PFCand);
    }
    if (passesSignalQualityCuts || passesIsolationQualityCuts) {
      selectedL1PFCandsSignalOrIsolationQualityCuts.push_back(l1PFCand);
    }
  }

  // sort PFCandidate collection by decreasing pT
  std::sort(selectedL1PFCandsSignalQualityCuts.begin(), selectedL1PFCandsSignalQualityCuts.end(), isHigherPt_pfCandRef);
  std::sort(selectedL1PFCandsSignalOrIsolationQualityCuts.begin(),
            selectedL1PFCandsSignalOrIsolationQualityCuts.end(),
            isHigherPt_pfCandRef);

  if (debug_) {
    std::cout << "AFTER selection (signalQualityCuts):" << std::endl;
    for (auto l1PFCand : selectedL1PFCandsSignalQualityCuts) {
      printPFCand(std::cout, *l1PFCand, primaryVertex_z);
    }
  }

  l1t::L1HPSPFTauCollection l1PFTauCollectionUncleaned;

  if (useChargedPFCandSeeds_) {
    for (auto l1PFCand : selectedL1PFCandsSignalQualityCuts) {
      if (l1PFCand->charge() != 0 && l1PFCand->pt() > minSeedChargedPFCandPt_ &&
          std::fabs(l1PFCand->eta()) < maxSeedChargedPFCandEta_) {
        bool isFromPrimaryVertex = false;
        if (primaryVertex.get()) {
          l1t::PFTrackRef l1PFTrack = l1PFCand->pfTrack();
          double dz = std::fabs(l1PFTrack->vertex().z() - primaryVertex_z);
          if (dz < maxSeedChargedPFCandDz_) {
            isFromPrimaryVertex = true;
          }
        } else {
          isFromPrimaryVertex = true;
        }
        if (isFromPrimaryVertex) {
          tauBuilder_->reset();
          tauBuilder_->setL1PFCandProductID(l1PFCands.id());
          tauBuilder_->setVertex(primaryVertex);
          tauBuilder_->setL1PFTauSeed(l1PFCand);
          tauBuilder_->addL1PFCandidates(selectedL1PFCandsSignalOrIsolationQualityCuts);
          tauBuilder_->buildL1PFTau();
          l1t::L1HPSPFTau l1PFTau = tauBuilder_->getL1PFTau();
          if (l1PFTau.pt() > 1.)
            l1PFTauCollectionUncleaned.push_back(l1PFTau);
        }
      }
    }
  }

  if (useJetSeeds_) {
    edm::Handle<std::vector<reco::CaloJet>> l1Jets;
    evt.getByToken(tokenL1Jets_, l1Jets);

    size_t numL1Jets = l1Jets->size();
    for (size_t idxL1Jet = 0; idxL1Jet < numL1Jets; ++idxL1Jet) {
      reco::CaloJetRef l1Jet(l1Jets, idxL1Jet);
      if (l1Jet->pt() > minSeedJetPt_ && std::fabs(l1Jet->eta()) < maxSeedJetEta_) {
        tauBuilder_->reset();
        tauBuilder_->setL1PFCandProductID(l1PFCands.id());
        tauBuilder_->setVertex(primaryVertex);
        //tauBuilder_->setL1PFTauSeed(l1Jet);
        tauBuilder_->setL1PFTauSeed(l1Jet, selectedL1PFCandsSignalQualityCuts);
        tauBuilder_->addL1PFCandidates(selectedL1PFCandsSignalOrIsolationQualityCuts);
        tauBuilder_->buildL1PFTau();
        l1t::L1HPSPFTau l1PFTau = tauBuilder_->getL1PFTau();
        if (l1PFTau.pt() > 1.)
          l1PFTauCollectionUncleaned.push_back(l1PFTau);
      }
    }
  }

  // sort PFTau candidate collection by decreasing pT
  std::sort(l1PFTauCollectionUncleaned.begin(), l1PFTauCollectionUncleaned.end(), isHigherPt_pfTau);

  if (debug_) {
    std::cout << "BEFORE cleaning:" << std::endl;
    for (size_t idx = 0; idx < l1PFTauCollectionUncleaned.size(); ++idx) {
      const l1t::L1HPSPFTau& l1PFTau = l1PFTauCollectionUncleaned.at(idx);
      std::cout << "L1HPSPFTau #" << idx << ": " << l1PFTau;
    }
  }

  for (auto l1PFTau : l1PFTauCollectionUncleaned) {
    if (applyPreselection_ &&
        !(l1PFTau.pt() > minPFTauPt_ && std::fabs(l1PFTau.eta()) < maxPFTauEta_ &&
          l1PFTau.leadChargedPFCand().isNonnull() && l1PFTau.leadChargedPFCand()->pt() > minLeadChargedPFCandPt_ &&
          std::fabs(l1PFTau.leadChargedPFCand()->eta()) < maxLeadChargedPFCandEta_ &&
          (srcL1Vertices_.label() == "" ||
           (primaryVertex.isNonnull() && l1PFTau.leadChargedPFCand()->pfTrack().isNonnull() &&
            std::fabs(l1PFTau.leadChargedPFCand()->pfTrack()->vertex().z() - primaryVertex->zvertex()) <
                maxLeadChargedPFCandDz_)) &&
          l1PFTau.sumChargedIso() < maxChargedIso_ && l1PFTau.sumChargedIso() < maxChargedRelIso_ * l1PFTau.pt()))
      continue;

    bool isOverlap = false;
    for (auto l1PFTau2 : *l1PFTauCollectionCleaned) {
      double deltaEta = l1PFTau.eta() - l1PFTau2.eta();
      double deltaPhi = l1PFTau.phi() - l1PFTau2.phi();
      if ((deltaEta * deltaEta + deltaPhi * deltaPhi) < deltaR2Cleaning_) {
        isOverlap = true;
      }
    }
    if (!isOverlap) {
      l1PFTauCollectionCleaned->push_back(l1PFTau);
    }
  }

  if (debug_) {
    std::cout << "AFTER cleaning:" << std::endl;
    for (size_t idx = 0; idx < l1PFTauCollectionCleaned->size(); ++idx) {
      const l1t::L1HPSPFTau& l1PFTau = l1PFTauCollectionCleaned->at(idx);
      std::cout << "L1HPSPFTau #" << idx << ": " << l1PFTau;
    }
  }

  evt.put(std::move(l1PFTauCollectionCleaned));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1HPSPFTauProducer);
