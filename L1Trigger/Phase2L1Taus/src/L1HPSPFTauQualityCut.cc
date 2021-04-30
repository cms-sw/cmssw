#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauQualityCut.h"
#include "FWCore/Utilities/interface/Exception.h"  // cms::Exception

L1HPSPFTauQualityCut::L1HPSPFTauQualityCut(const edm::ParameterSet& cfg)
    : debug_(cfg.getUntrackedParameter<bool>("debug", false)) {
  std::string pfCandTypeString = cfg.getParameter<std::string>("pfCandType");
  if (pfCandTypeString == "chargedHadron")
    pfCandType_ = l1t::PFCandidate::ChargedHadron;
  else if (pfCandTypeString == "electron")
    pfCandType_ = l1t::PFCandidate::Electron;
  else if (pfCandTypeString == "muon")
    pfCandType_ = l1t::PFCandidate::Muon;
  else if (pfCandTypeString == "neutralHadron")
    pfCandType_ = l1t::PFCandidate::NeutralHadron;
  else if (pfCandTypeString == "photon")
    pfCandType_ = l1t::PFCandidate::Photon;
  else
    throw cms::Exception("L1HPSPFTauQualityCut")
        << "Invalid Configuration parameter 'pfCandType' = '" << pfCandTypeString << "' !!\n";

  std::string dzCutString = cfg.getParameter<std::string>("dzCut");
  if (dzCutString == "disabled")
    dzCut_ = kDisabled;
  else if (dzCutString == "enabled_primary")
    dzCut_ = kEnabledPrimary;
  else if (dzCutString == "enabled_pileup")
    dzCut_ = kEnabledPileup;
  else
    throw cms::Exception("L1HPSPFTauQualityCut")
        << "Invalid Configuration parameter 'dzCut' = '" << dzCutString << "' !!\n";

  minPt_ = cfg.getParameter<double>("minPt");
  maxDz_ = (cfg.exists("maxDz")) ? cfg.getParameter<double>("maxDz") : 1.e+3;

  if (debug_ && dzCut_ == kEnabledPrimary) {
    std::cout << " applying pT > " << minPt_ << " GeV && dz < " << maxDz_ << " cm to PFCands of type = '"
              << pfCandTypeString << "'" << std::endl;
  }
}

bool L1HPSPFTauQualityCut::operator()(const l1t::PFCandidate& pfCand, float_t primaryVertex_z) const {
  if (pfCand.id() == pfCandType_) {
    if (pfCand.pt() < minPt_) {
      return false;
    }

    if (pfCand.charge() != 0) {
      if (dzCut_ == kEnabledPrimary || dzCut_ == kEnabledPileup) {
        l1t::PFTrackRef pfCand_track = pfCand.pfTrack();
        double dz = std::fabs(pfCand_track->vertex().z() - primaryVertex_z);
        if (dzCut_ == kEnabledPrimary && dz > maxDz_)
          return false;
        if (dzCut_ == kEnabledPileup && dz <= maxDz_)
          return false;
      }
    } else if (dzCut_ == kEnabledPileup) {
      return false;  // CV: only consider charged PFCands as originating from pileup
    }
  }
  return true;
}

l1t::PFCandidate::ParticleType L1HPSPFTauQualityCut::pfCandType() const { return pfCandType_; }

int L1HPSPFTauQualityCut::dzCut() const { return dzCut_; }

float_t L1HPSPFTauQualityCut::minPt() const { return minPt_; }

float_t L1HPSPFTauQualityCut::maxDz() const { return maxDz_; }

L1HPSPFTauQualityCut readL1PFTauQualityCut(const edm::ParameterSet& cfg,
                                           const std::string& pfCandType,
                                           const std::string& dzCut,
                                           bool debug) {
  edm::ParameterSet cfg_pfCandType = cfg.getParameter<edm::ParameterSet>(pfCandType);
  cfg_pfCandType.addParameter<std::string>("pfCandType", pfCandType);
  cfg_pfCandType.addParameter<std::string>("dzCut", dzCut);
  cfg_pfCandType.addUntrackedParameter<bool>("debug", debug);
  L1HPSPFTauQualityCut qualityCut(cfg_pfCandType);
  return qualityCut;
}

std::vector<L1HPSPFTauQualityCut> readL1PFTauQualityCuts(const edm::ParameterSet& cfg,
                                                         const std::string& dzCut,
                                                         bool debug) {
  std::vector<L1HPSPFTauQualityCut> qualityCuts;
  qualityCuts.push_back(readL1PFTauQualityCut(cfg, "chargedHadron", dzCut, debug));
  qualityCuts.push_back(readL1PFTauQualityCut(cfg, "electron", dzCut, debug));
  qualityCuts.push_back(readL1PFTauQualityCut(cfg, "muon", dzCut, debug));
  qualityCuts.push_back(readL1PFTauQualityCut(cfg, "photon", dzCut, debug));
  qualityCuts.push_back(readL1PFTauQualityCut(cfg, "neutralHadron", dzCut, debug));
  return qualityCuts;
}

bool isSelected(const std::vector<L1HPSPFTauQualityCut>& qualityCuts,
                const l1t::PFCandidate& pfCand,
                float_t primaryVertex_z) {
  for (auto qualityCut : qualityCuts) {
    if (!qualityCut(pfCand, primaryVertex_z))
      return false;
  }
  return true;
}
