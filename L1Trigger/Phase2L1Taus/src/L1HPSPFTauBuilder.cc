#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauBuilder.h"
#include "FWCore/Utilities/interface/Exception.h"  // cms::Exception
#include "DataFormats/Math/interface/deltaR.h"     // reco::deltaR
#include <regex>                                   // sd::regex_replace
#include <TMath.h>                                 // TMath::Pi()
#include <string>                                  // std::string
#include <algorithm>                               // std::max(), std::sort()
#include <cmath>                                   // std::fabs

namespace {
  std::string getSignalConeSizeFormula(const edm::ParameterSet& cfg) {
    return std::regex_replace(cfg.getParameter<std::string>("signalConeSize"), std::regex("pt"), "x");
  }
}  // namespace

L1HPSPFTauBuilder::L1HPSPFTauBuilder(const edm::ParameterSet& cfg)
    : signalConeSizeFormula_(getSignalConeSizeFormula(cfg)),
      minSignalConeSize_(cfg.getParameter<double>("minSignalConeSize")),
      maxSignalConeSize_(cfg.getParameter<double>("maxSignalConeSize")),
      useStrips_(cfg.getParameter<bool>("useStrips")),
      stripSizeEta_(cfg.getParameter<double>("stripSizeEta")),
      stripSizePhi_(cfg.getParameter<double>("stripSizePhi")),
      isolationConeSize_(cfg.getParameter<double>("isolationConeSize")),
      debug_(cfg.getUntrackedParameter<bool>("debug", false)) {
  assert(maxSignalConeSize_ >= minSignalConeSize_);

  isolationConeSize2_ = isolationConeSize_ * isolationConeSize_;

  if (debug_) {
    std::cout << "setting Quality cuts for signal PFCands:" << std::endl;
  }
  edm::ParameterSet cfg_signalQualityCuts = cfg.getParameter<edm::ParameterSet>("signalQualityCuts");
  signalQualityCutsDzCutDisabled_ = readL1PFTauQualityCuts(cfg_signalQualityCuts, "disabled", debug_);
  signalQualityCutsDzCutEnabledPrimary_ = readL1PFTauQualityCuts(cfg_signalQualityCuts, "enabled_primary", debug_);
  if (debug_) {
    std::cout << "setting Quality cuts for isolation PFCands:" << std::endl;
  }
  edm::ParameterSet cfg_isolationQualityCuts = cfg.getParameter<edm::ParameterSet>("isolationQualityCuts");
  isolationQualityCutsDzCutDisabled_ = readL1PFTauQualityCuts(cfg_isolationQualityCuts, "disabled", debug_);
  isolationQualityCutsDzCutEnabledPrimary_ =
      readL1PFTauQualityCuts(cfg_isolationQualityCuts, "enabled_primary", debug_);
  isolationQualityCutsDzCutEnabledPileup_ = readL1PFTauQualityCuts(cfg_isolationQualityCuts, "enabled_pileup", debug_);
}

void L1HPSPFTauBuilder::reset() {
  signalConeSize_ = 0.;
  signalConeSize2_ = 0.;

  l1PFCandProductID_ = edm::ProductID();
  isPFCandSeeded_ = false;
  l1PFCandSeed_ = l1t::PFCandidateRef();
  isJetSeeded_ = false;
  l1JetSeed_ = reco::CaloJetRef();
  l1PFTauSeedEta_ = 0.;
  l1PFTauSeedPhi_ = 0.;
  l1PFTauSeedZVtx_ = 0.;
  sumAllL1PFCandidatesPt_ = 0.;
  primaryVertex_ = l1t::VertexWordRef();
  l1PFTau_ = l1t::HPSPFTau();

  stripP4_ = reco::Particle::LorentzVector(0., 0., 0., 0.);

  signalAllL1PFCandidates_.clear();
  signalChargedHadrons_.clear();
  signalElectrons_.clear();
  signalNeutralHadrons_.clear();
  signalPhotons_.clear();
  signalMuons_.clear();

  stripAllL1PFCandidates_.clear();
  stripElectrons_.clear();
  stripPhotons_.clear();

  isoAllL1PFCandidates_.clear();
  isoChargedHadrons_.clear();
  isoElectrons_.clear();
  isoNeutralHadrons_.clear();
  isoPhotons_.clear();
  isoMuons_.clear();

  sumAllL1PFCandidates_.clear();
  sumChargedHadrons_.clear();
  sumElectrons_.clear();
  sumNeutralHadrons_.clear();
  sumPhotons_.clear();
  sumMuons_.clear();

  sumChargedIsoPileup_ = 0.;
}

void L1HPSPFTauBuilder::setL1PFCandProductID(const edm::ProductID& l1PFCandProductID) {
  l1PFCandProductID_ = l1PFCandProductID;
}

void L1HPSPFTauBuilder::setVertex(const l1t::VertexWordRef& primaryVertex) { primaryVertex_ = primaryVertex; }

void L1HPSPFTauBuilder::setL1PFTauSeed(const l1t::PFCandidateRef& l1PFCandSeed) {
  if (debug_) {
    std::cout << "<L1HPSPFTauBuilder::setL1PFTauSeed>:" << std::endl;
    std::cout << "seeding HPSPFTau with ChargedPFCand:";
    printPFCand(std::cout, *l1PFCandSeed, primaryVertex_);
  }

  l1PFCandSeed_ = l1PFCandSeed;
  l1PFTauSeedEta_ = l1PFCandSeed->eta();
  l1PFTauSeedPhi_ = l1PFCandSeed->phi();
  if (l1PFCandSeed->charge() != 0 && l1PFCandSeed->pfTrack().isNonnull()) {
    l1PFTauSeedZVtx_ = l1PFCandSeed->pfTrack()->vertex().z();
    isPFCandSeeded_ = true;
  }
}
// This is commented as l1JetSeed->numberOfDaughters() = 0
// Alternative way is used below for the moment
/* 
void L1HPSPFTauBuilder::setL1PFTauSeed(const reco::CaloJetRef& l1JetSeed) {
  if (debug_) {
    std::cout << "<L1HPSPFTauBuilder::setL1PFTauSeed>:" << std::endl;
    std::cout << "seeding HPSPFTau with Jet:";
    std::cout << " pT = " << l1JetSeed->pt() << ", eta = " << l1JetSeed->eta() << ", phi = " << l1JetSeed->phi()
              << std::endl;
  }

  l1JetSeed_ = l1JetSeed;
  reco::Candidate::LorentzVector l1PFTauSeed_p4;
  float l1PFTauSeedZVtx = 0.;
  bool l1PFTauSeed_hasVtx = false;
  float max_chargedPFCand_pt = -1.;
  size_t numConstituents = l1JetSeed->numberOfDaughters();
  for (size_t idxConstituent = 0; idxConstituent < numConstituents; ++idxConstituent) {
    const l1t::PFCandidate* l1PFCand = dynamic_cast<const l1t::PFCandidate*>(l1JetSeed->daughter(idxConstituent));
    if (!l1PFCand) {
      throw cms::Exception("L1HPSPFTauBuilder") << "Jet was not built from l1t::PFCandidates !!\n";
    }
    if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron || l1PFCand->id() == l1t::PFCandidate::Electron ||
        l1PFCand->id() == l1t::PFCandidate::Photon || l1PFCand->id() == l1t::PFCandidate::Muon) {
      l1PFTauSeed_p4 += l1PFCand->p4();
      if (l1PFCand->charge() != 0 && l1PFCand->pfTrack().isNonnull() && l1PFCand->pt() > max_chargedPFCand_pt) {
        l1PFTauSeedZVtx = l1PFCand->pfTrack()->vertex().z();
        l1PFTauSeed_hasVtx = true;
        max_chargedPFCand_pt = l1PFCand->pt();
      }
    }
  }
  if (l1PFTauSeed_p4.pt() > 1. && l1PFTauSeed_hasVtx) {
    l1PFTauSeedEta_ = l1PFTauSeed_p4.eta();
    l1PFTauSeedPhi_ = l1PFTauSeed_p4.phi();
    l1PFTauSeedZVtx_ = l1PFTauSeedZVtx;
    isJetSeeded_ = true;
  }
}
*/
void L1HPSPFTauBuilder::setL1PFTauSeed(const reco::CaloJetRef& l1JetSeed,
                                       const std::vector<l1t::PFCandidateRef>& l1PFCands) {
  if (debug_) {
    std::cout << "<L1HPSPFTauBuilder::setL1PFTauSeed>:" << std::endl;
    std::cout << "seeding HPSPFTau with Jet:";
    std::cout << " pT = " << l1JetSeed->pt() << ", eta = " << l1JetSeed->eta() << ", phi = " << l1JetSeed->phi()
              << std::endl;
  }

  l1JetSeed_ = l1JetSeed;
  reco::Candidate::LorentzVector l1PFTauSeed_p4;
  float l1PFTauSeedZVtx = 0.;
  bool l1PFTauSeed_hasVtx = false;
  float max_chargedPFCand_pt = -1.;
  for (const auto& l1PFCand : l1PFCands) {
    double dR = reco::deltaR(l1PFCand->eta(), l1PFCand->phi(), l1JetSeed->eta(), l1JetSeed->phi());
    if (dR > 0.4)
      continue;
    if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron || l1PFCand->id() == l1t::PFCandidate::Electron ||
        l1PFCand->id() == l1t::PFCandidate::Photon || l1PFCand->id() == l1t::PFCandidate::Muon) {
      l1PFTauSeed_p4 += l1PFCand->p4();
      if (l1PFCand->charge() != 0 && l1PFCand->pfTrack().isNonnull() && l1PFCand->pt() > max_chargedPFCand_pt) {
        l1PFTauSeedZVtx = l1PFCand->pfTrack()->vertex().z();
        l1PFTauSeed_hasVtx = true;
        max_chargedPFCand_pt = l1PFCand->pt();
      }
    }
  }
  if (l1PFTauSeed_p4.pt() > 1. && l1PFTauSeed_hasVtx) {
    l1PFTauSeedEta_ = l1PFTauSeed_p4.eta();
    l1PFTauSeedPhi_ = l1PFTauSeed_p4.phi();
    l1PFTauSeedZVtx_ = l1PFTauSeedZVtx;
    isJetSeeded_ = true;
  }
}

void L1HPSPFTauBuilder::addL1PFCandidates(const std::vector<l1t::PFCandidateRef>& l1PFCands) {
  if (debug_) {
    std::cout << "<L1HPSPFTauBuilder::addL1PFCandidates>:" << std::endl;
  }

  // do not build tau candidates for which no reference z-position exists,
  // as in this case charged PFCands originating from the primary (hard-scatter) interaction
  // cannot be distinguished from charged PFCands originating from pileup
  if (!(isPFCandSeeded_ || isJetSeeded_))
    return;

  for (const auto& l1PFCand : l1PFCands) {
    if (!isWithinIsolationCone(*l1PFCand))
      continue;
    sumAllL1PFCandidates_.push_back(l1PFCand);
    if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron) {
      sumChargedHadrons_.push_back(l1PFCand);
    } else if (l1PFCand->id() == l1t::PFCandidate::Electron) {
      sumElectrons_.push_back(l1PFCand);
    } else if (l1PFCand->id() == l1t::PFCandidate::NeutralHadron) {
      sumNeutralHadrons_.push_back(l1PFCand);
    } else if (l1PFCand->id() == l1t::PFCandidate::Photon) {
      sumPhotons_.push_back(l1PFCand);
    } else if (l1PFCand->id() == l1t::PFCandidate::Muon) {
      sumMuons_.push_back(l1PFCand);
    }
  }

  for (const auto& l1PFCand : sumAllL1PFCandidates_) {
    sumAllL1PFCandidatesPt_ += l1PFCand->pt();
  }
  std::vector<double> emptyV;
  std::vector<double> sumAllL1PFCandidatesPt(1);
  sumAllL1PFCandidatesPt[0] = sumAllL1PFCandidatesPt_;

  signalConeSize_ = signalConeSizeFormula_.evaluate(sumAllL1PFCandidatesPt, emptyV);

  if (signalConeSize_ < minSignalConeSize_)
    signalConeSize_ = minSignalConeSize_;
  if (signalConeSize_ > maxSignalConeSize_)
    signalConeSize_ = maxSignalConeSize_;
  signalConeSize2_ = signalConeSize_ * signalConeSize_;

  for (const auto& l1PFCand : sumAllL1PFCandidates_) {
    if (debug_) {
      printPFCand(std::cout, *l1PFCand, primaryVertex_);
    }

    bool isSignalPFCand = false;
    bool isStripPFCand = false;
    bool isElectron_or_Photon =
        l1PFCand->id() == l1t::PFCandidate::Electron || l1PFCand->id() == l1t::PFCandidate::Photon;
    bool isChargedHadron = l1PFCand->id() == l1t::PFCandidate::ChargedHadron;
    if (isWithinSignalCone(*l1PFCand) && !(isChargedHadron && signalChargedHadrons_.size() > 3)) {
      isSignalPFCand = true;
    }
    if (isElectron_or_Photon && isWithinStrip(*l1PFCand)) {
      if (useStrips_) {
        isSignalPFCand = true;
      }
      isStripPFCand = true;
    }
    bool passesSignalQualityCuts = isSelected(signalQualityCutsDzCutEnabledPrimary_, *l1PFCand, l1PFTauSeedZVtx_);
    if (isSignalPFCand && passesSignalQualityCuts) {
      signalAllL1PFCandidates_.push_back(l1PFCand);
      if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron) {
        signalChargedHadrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Electron) {
        signalElectrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::NeutralHadron) {
        signalNeutralHadrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Photon) {
        signalPhotons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Muon) {
        signalMuons_.push_back(l1PFCand);
      }
    }
    if (isStripPFCand && passesSignalQualityCuts) {
      stripAllL1PFCandidates_.push_back(l1PFCand);
      if (l1PFCand->id() == l1t::PFCandidate::Electron) {
        stripElectrons_.push_back(l1PFCand);
        stripP4_ += l1PFCand->p4();
      } else if (l1PFCand->id() == l1t::PFCandidate::Photon) {
        stripPhotons_.push_back(l1PFCand);
        stripP4_ += l1PFCand->p4();
      } else
        assert(0);
    }

    bool isIsolationPFCand = isWithinIsolationCone(*l1PFCand) && !isSignalPFCand;
    bool passesIsolationQualityCuts = isSelected(isolationQualityCutsDzCutEnabledPrimary_, *l1PFCand, l1PFTauSeedZVtx_);
    if (isIsolationPFCand && passesIsolationQualityCuts) {
      isoAllL1PFCandidates_.push_back(l1PFCand);
      if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron) {
        isoChargedHadrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Electron) {
        isoElectrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::NeutralHadron) {
        isoNeutralHadrons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Photon) {
        isoPhotons_.push_back(l1PFCand);
      } else if (l1PFCand->id() == l1t::PFCandidate::Muon) {
        isoMuons_.push_back(l1PFCand);
      }
    }

    if (debug_) {
      std::cout << "dR = " << reco::deltaR(l1PFCand->eta(), l1PFCand->phi(), l1PFTauSeedEta_, l1PFTauSeedPhi_) << ":"
                << " isSignalPFCand = " << isSignalPFCand << ", isStripPFCand = " << isStripPFCand
                << " (passesSignalQualityCuts = " << passesSignalQualityCuts << "),"
                << " isIsolationPFCand = " << isIsolationPFCand
                << " (passesIsolationQualityCuts = " << passesIsolationQualityCuts << ")" << std::endl;
    }
  }

  for (const auto& l1PFCand : l1PFCands) {
    if (!isWithinIsolationCone(*l1PFCand))
      continue;

    if (l1PFCand->charge() != 0 && isSelected(isolationQualityCutsDzCutEnabledPileup_, *l1PFCand, l1PFTauSeedZVtx_)) {
      sumChargedIsoPileup_ += l1PFCand->pt();
    }
  }
}

//void L1HPSPFTauBuilder::setRho(double rho) { rho_ = rho; }

bool L1HPSPFTauBuilder::isWithinSignalCone(const l1t::PFCandidate& l1PFCand) {
  if (isPFCandSeeded_ || isJetSeeded_) {
    double deltaEta = l1PFCand.eta() - l1PFTauSeedEta_;
    double deltaPhi = l1PFCand.phi() - l1PFTauSeedPhi_;
    if ((deltaEta * deltaEta + deltaPhi * deltaPhi) < signalConeSize2_)
      return true;
  }
  return false;
}

bool L1HPSPFTauBuilder::isWithinStrip(const l1t::PFCandidate& l1PFCand) {
  if (isPFCandSeeded_ || isJetSeeded_) {
    double deltaEta = l1PFCand.eta() - l1PFTauSeedEta_;
    double deltaPhi = l1PFCand.phi() - l1PFTauSeedPhi_;
    if (std::fabs(deltaEta) < stripSizeEta_ && std::fabs(deltaPhi) < stripSizePhi_)
      return true;
  }
  return false;
}

bool L1HPSPFTauBuilder::isWithinIsolationCone(const l1t::PFCandidate& l1PFCand) {
  double deltaEta = l1PFCand.eta() - l1PFTauSeedEta_;
  double deltaPhi = l1PFCand.phi() - l1PFTauSeedPhi_;
  if ((deltaEta * deltaEta + deltaPhi * deltaPhi) < isolationConeSize2_)
    return true;
  else
    return false;
}

void L1HPSPFTauBuilder::buildL1PFTau() {
  reco::Particle::LorentzVector l1PFTau_p4;
  for (const auto& l1PFCand : signalAllL1PFCandidates_) {
    if (l1PFCand->id() == l1t::PFCandidate::ChargedHadron || l1PFCand->id() == l1t::PFCandidate::Electron ||
        l1PFCand->id() == l1t::PFCandidate::Photon) {
      l1PFTau_p4 += l1PFCand->p4();
      if (l1PFCand->charge() != 0 &&
          (l1PFTau_.leadChargedPFCand().isNull() || l1PFCand->pt() > l1PFTau_.leadChargedPFCand()->pt())) {
        l1PFTau_.setLeadChargedPFCand(l1PFCand);
      }
    }
  }
  if (l1PFTau_.leadChargedPFCand().isNonnull() && l1PFTau_.leadChargedPFCand()->pfTrack().isNonnull()) {
    l1PFTau_.setZ(l1PFTau_.leadChargedPFCand()->pfTrack()->vertex().z());

    l1PFTau_.setP4(l1PFTau_p4);

    l1PFTau_.setSeedChargedPFCand(l1PFCandSeed_);
    l1PFTau_.setSeedJet(l1JetSeed_);

    l1PFTau_.setSignalAllL1PFCandidates(convertToRefVector(signalAllL1PFCandidates_));
    l1PFTau_.setSignalChargedHadrons(convertToRefVector(signalChargedHadrons_));
    l1PFTau_.setSignalElectrons(convertToRefVector(signalElectrons_));
    l1PFTau_.setSignalNeutralHadrons(convertToRefVector(signalNeutralHadrons_));
    l1PFTau_.setSignalPhotons(convertToRefVector(signalPhotons_));
    l1PFTau_.setSignalMuons(convertToRefVector(signalMuons_));

    l1PFTau_.setStripAllL1PFCandidates(convertToRefVector(stripAllL1PFCandidates_));
    l1PFTau_.setStripElectrons(convertToRefVector(stripElectrons_));
    l1PFTau_.setStripPhotons(convertToRefVector(stripPhotons_));

    l1PFTau_.setIsoAllL1PFCandidates(convertToRefVector(isoAllL1PFCandidates_));
    l1PFTau_.setIsoChargedHadrons(convertToRefVector(isoChargedHadrons_));
    l1PFTau_.setIsoElectrons(convertToRefVector(isoElectrons_));
    l1PFTau_.setIsoNeutralHadrons(convertToRefVector(isoNeutralHadrons_));
    l1PFTau_.setIsoPhotons(convertToRefVector(isoPhotons_));
    l1PFTau_.setIsoMuons(convertToRefVector(isoMuons_));

    l1PFTau_.setSumAllL1PFCandidates(convertToRefVector(sumAllL1PFCandidates_));
    l1PFTau_.setSumChargedHadrons(convertToRefVector(sumChargedHadrons_));
    l1PFTau_.setSumElectrons(convertToRefVector(sumElectrons_));
    l1PFTau_.setSumNeutralHadrons(convertToRefVector(sumNeutralHadrons_));
    l1PFTau_.setSumPhotons(convertToRefVector(sumPhotons_));
    l1PFTau_.setSumMuons(convertToRefVector(sumMuons_));

    l1PFTau_.setPrimaryVertex(primaryVertex_);

    if (l1PFTau_.signalChargedHadrons().size() > 1) {
      if (stripP4_.pt() < 5.)
        l1PFTau_.setTauType(l1t::HPSPFTau::kThreeProng0Pi0);
      else
        l1PFTau_.setTauType(l1t::HPSPFTau::kThreeProng1Pi0);
    } else {
      if (stripP4_.pt() < 5.)
        l1PFTau_.setTauType(l1t::HPSPFTau::kOneProng0Pi0);
      else
        l1PFTau_.setTauType(l1t::HPSPFTau::kOneProng1Pi0);
    }

    l1PFTau_.setStripP4(stripP4_);

    l1PFTau_.setSumAllL1PFCandidatesPt(sumAllL1PFCandidatesPt_);
    l1PFTau_.setSignalConeSize(signalConeSize_);
    l1PFTau_.setisolationConeSize(isolationConeSize_);

    double sumChargedIso = 0.;
    double sumNeutralIso = 0.;
    for (const auto& l1PFCand : isoAllL1PFCandidates_) {
      if (l1PFCand->charge() != 0) {
        sumChargedIso += l1PFCand->pt();
      } else if (l1PFCand->id() == l1t::PFCandidate::Photon) {
        sumNeutralIso += l1PFCand->pt();
      }
    }
    l1PFTau_.setSumChargedIso(sumChargedIso);
    l1PFTau_.setSumNeutralIso(sumNeutralIso);
    const double weightNeutralIso = 1.;
    const double offsetNeutralIso = 0.;
    l1PFTau_.setSumCombinedIso(sumChargedIso + weightNeutralIso * (sumNeutralIso - offsetNeutralIso));
    l1PFTau_.setSumChargedIsoPileup(sumChargedIsoPileup_);

    if (l1PFTau_.sumChargedIso() < 20.0) {
      l1PFTau_.setPassVLooseIso(true);
    }
    if (l1PFTau_.sumChargedIso() < 10.0) {
      l1PFTau_.setPassLooseIso(true);
    }
    if (l1PFTau_.sumChargedIso() < 5.0) {
      l1PFTau_.setPassMediumIso(true);
    }
    if (l1PFTau_.sumChargedIso() < 2.5) {
      l1PFTau_.setPassTightIso(true);
    }

    if (l1PFTau_p4.pt() != 0) {
      if (l1PFTau_.sumChargedIso() / l1PFTau_p4.pt() < 0.40) {
        l1PFTau_.setPassVLooseRelIso(true);
      }
      if (l1PFTau_.sumChargedIso() / l1PFTau_p4.pt() < 0.20) {
        l1PFTau_.setPassLooseRelIso(true);
      }
      if (l1PFTau_.sumChargedIso() / l1PFTau_p4.pt() < 0.10) {
        l1PFTau_.setPassMediumRelIso(true);
      }
      if (l1PFTau_.sumChargedIso() / l1PFTau_p4.pt() < 0.05) {
        l1PFTau_.setPassTightRelIso(true);
      }
    }
  }
}

l1t::PFCandidateRefVector L1HPSPFTauBuilder::convertToRefVector(const std::vector<l1t::PFCandidateRef>& l1PFCands) {
  l1t::PFCandidateRefVector l1PFCandsRefVector(l1PFCandProductID_);
  for (const auto& l1PFCand : l1PFCands) {
    l1PFCandsRefVector.push_back(l1PFCand);
  }
  return l1PFCandsRefVector;
}
