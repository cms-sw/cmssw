#include "L1Trigger/Phase2L1ParticleFlow/interface/PuppiAlgo.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include "Math/ProbFunc.h"

namespace {
  std::vector<float> vd2vf(const std::vector<double> &vd) {
    std::vector<float> ret;
    ret.insert(ret.end(), vd.begin(), vd.end());
    return ret;
  }
}  // namespace

using namespace l1tpf_impl;

PuppiAlgo::PuppiAlgo(const edm::ParameterSet &iConfig)
    : PUAlgoBase(iConfig),
      puppiDr_(iConfig.getParameter<double>("puppiDr")),
      puppiDrMin_(iConfig.getParameter<double>("puppiDrMin")),
      puppiPtMax_(iConfig.getParameter<double>("puppiPtMax")),
      puppiEtaCuts_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiEtaCuts"))),
      puppiPtCuts_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtCuts"))),
      puppiPtCutsPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtCutsPhotons"))),
      puppiUsingBareTracks_(iConfig.getParameter<bool>("puppiUsingBareTracks")) {
  debug_ = iConfig.getUntrackedParameter<int>("puppiDebug", debug_);
  if (puppiEtaCuts_.size() != puppiPtCuts_.size() || puppiPtCuts_.size() != puppiPtCutsPhotons_.size()) {
    throw cms::Exception("Configuration", "Bad PUPPI config");
  }
  for (unsigned int i = 0, n = puppiEtaCuts_.size(); i < n; ++i) {
    intPuppiEtaCuts_.push_back(std::round(puppiEtaCuts_[i] * CaloCluster::ETAPHI_SCALE));
    intPuppiPtCuts_.push_back(std::round(puppiPtCuts_[i] * CaloCluster::PT_SCALE));
    intPuppiPtCutsPhotons_.push_back(std::round(puppiPtCutsPhotons_[i] * CaloCluster::PT_SCALE));
  }
}

PuppiAlgo::~PuppiAlgo() {}

const std::vector<std::string> &PuppiAlgo::puGlobalNames() const {
  static const std::vector<std::string> names_{"alphaCMed", "alphaCRms", "alphaFMed", "alphaFRms"};
  return names_;
}
void PuppiAlgo::doPUGlobals(const std::vector<Region> &rs, float z0, float npu, std::vector<float> &globals) const {
  globals.resize(4);
  computePuppiMedRMS(rs, globals[0], globals[1], globals[2], globals[3]);
}

void PuppiAlgo::runNeutralsPU(Region &r, float z0, float npu, const std::vector<float> &globals) const {
  std::vector<float> alphaC, alphaF;
  computePuppiAlphas(r, alphaC, alphaF);
  computePuppiWeights(r, alphaC, alphaF, globals[0], globals[1], globals[2], globals[3]);
  fillPuppi(r);
}

void PuppiAlgo::runNeutralsPU(Region &r, std::vector<float> &z0, float npu, const std::vector<float> &globals) const {
  float z0tmp = 0;
  runNeutralsPU(r, z0tmp, npu, globals);
}

void PuppiAlgo::computePuppiAlphas(const Region &r, std::vector<float> &alphaC, std::vector<float> &alphaF) const {
  alphaC.resize(r.pf.size());
  alphaF.resize(r.pf.size());
  float puppiDr2 = std::pow(puppiDr_, 2), puppiDr2min = std::pow(puppiDrMin_, 2);
  for (unsigned int ip = 0, np = r.pf.size(); ip < np; ++ip) {
    const PFParticle &p = r.pf[ip];
    if (p.hwId <= 1)
      continue;
    // neutral
    alphaC[ip] = 0;
    alphaF[ip] = 0;
    for (const PFParticle &p2 : r.pf) {
      float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
      if (dr2 > 0 && dr2 < puppiDr2) {
        float w = std::pow(std::min(p2.floatPt(), puppiPtMax_), 2) / std::max<float>(puppiDr2min, dr2);
        alphaF[ip] += w;
        if (p2.chargedPV)
          alphaC[ip] += w;
      }
    }
    if (puppiUsingBareTracks_) {
      alphaC[ip] = 0;
      for (const PropagatedTrack &p2 : r.track) {
        if (!p2.fromPV)
          continue;
        if (!p2.quality(l1tpf_impl::InputTrack::PFLOOSE))
          continue;
        float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
        if (dr2 > 0 && dr2 < puppiDr2) {
          alphaC[ip] += std::pow(std::min(p2.floatPt(), puppiPtMax_), 2) / std::max<float>(puppiDr2min, dr2);
        }
      }
    }
  }
}

void PuppiAlgo::computePuppiWeights(Region &r,
                                    const std::vector<float> &alphaC,
                                    const std::vector<float> &alphaF,
                                    float alphaCMed,
                                    float alphaCRms,
                                    float alphaFMed,
                                    float alphaFRms) const {
  int16_t ietacut = std::round(etaCharged_ * CaloCluster::ETAPHI_SCALE);
  for (unsigned int ip = 0, np = r.pf.size(); ip < np; ++ip) {
    PFParticle &p = r.pf[ip];
    // charged
    if (p.hwId == l1t::PFCandidate::ChargedHadron || p.hwId == l1t::PFCandidate::Electron ||
        p.hwId == l1t::PFCandidate::Muon) {
      p.setPuppiW(p.chargedPV || p.hwId == l1t::PFCandidate::Muon ? 1.0 : 0);
      if (debug_)
        dbgPrintf(
            "PUPPI \t charged id %1d pt %7.2f eta %+5.2f phi %+5.2f  alpha %+7.2f x2 %+7.2f --> puppi weight %.3f   "
            "puppi pt %7.2f \n",
            p.hwId,
            p.floatPt(),
            p.floatEta(),
            p.floatPhi(),
            0.,
            0.,
            p.floatPuppiW(),
            p.floatPt() * p.floatPuppiW());
      continue;
    }
    // neutral
    float alpha = -99, x2 = -99;
    bool central = std::abs(p.hwEta) < ietacut;
    if (r.relativeCoordinates)
      central =
          (std::abs(r.globalAbsEta(p.floatEta())) < etaCharged_);  // FIXME could make a better integer implementation
    if (central) {
      if (alphaC[ip] > 0) {
        alpha = std::log(alphaC[ip]);
        x2 = (alpha - alphaCMed) * std::abs(alpha - alphaCMed) / std::pow(alphaCRms, 2);
        p.setPuppiW(ROOT::Math::chisquared_cdf(x2, 1));
      } else {
        p.setPuppiW(0);
      }
    } else {
      if (alphaF[ip] > 0) {
        alpha = std::log(alphaF[ip]);
        x2 = (alpha - alphaFMed) * std::abs(alpha - alphaFMed) / std::pow(alphaFRms, 2);
        p.setPuppiW(ROOT::Math::chisquared_cdf(x2, 1));
      } else {
        p.setPuppiW(0);
      }
    }
    if (debug_)
      dbgPrintf(
          "PUPPI \t neutral id %1d pt %7.2f eta %+5.2f phi %+5.2f  alpha %+7.2f x2 %+7.2f --> puppi weight %.3f   "
          "puppi pt %7.2f \n",
          p.hwId,
          p.floatPt(),
          p.floatEta(),
          p.floatPhi(),
          alpha,
          x2,
          p.floatPuppiW(),
          p.floatPt() * p.floatPuppiW());
  }
}

void PuppiAlgo::computePuppiMedRMS(
    const std::vector<Region> &rs, float &alphaCMed, float &alphaCRms, float &alphaFMed, float &alphaFRms) const {
  std::vector<float> alphaFs;
  std::vector<float> alphaCs;
  int16_t ietacut = std::round(etaCharged_ * CaloCluster::ETAPHI_SCALE);
  float puppiDr2 = std::pow(puppiDr_, 2), puppiDr2min = std::pow(puppiDrMin_, 2);
  for (const Region &r : rs) {
    for (const PFParticle &p : r.pf) {
      bool central = std::abs(p.hwEta) < ietacut;
      if (r.relativeCoordinates)
        central = (r.globalAbsEta(p.floatEta()) < etaCharged_);  // FIXME could make a better integer implementation
      if (central) {
        if (p.hwId > 1 || p.chargedPV)
          continue;
      }
      float alphaC = 0, alphaF = 0;
      for (const PFParticle &p2 : r.pf) {
        float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
        if (dr2 > 0 && dr2 < puppiDr2) {
          float w = std::pow(std::min(p2.floatPt(), puppiPtMax_), 2) / std::max<float>(puppiDr2min, dr2);
          alphaF += w;
          if (p2.chargedPV)
            alphaC += w;
        }
      }
      if (puppiUsingBareTracks_) {
        alphaC = 0;
        for (const PropagatedTrack &p2 : r.track) {
          if (!p2.fromPV)
            continue;
          float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
          if (dr2 > 0 && dr2 < puppiDr2) {
            alphaC += std::pow(std::min(p2.floatPt(), puppiPtMax_), 2) / std::max<float>(puppiDr2min, dr2);
          }
        }
      }
      if (central) {
        if (alphaC > 0)
          alphaCs.push_back(std::log(alphaC));
      } else {
        if (alphaF > 0)
          alphaFs.push_back(std::log(alphaF));
      }
    }
  }
  std::sort(alphaCs.begin(), alphaCs.end());
  std::sort(alphaFs.begin(), alphaFs.end());

  if (alphaCs.size() > 1) {
    alphaCMed = alphaCs[alphaCs.size() / 2 + 1];
    double sum = 0.0;
    for (float alpha : alphaCs)
      sum += std::pow(alpha - alphaCMed, 2);
    alphaCRms = std::sqrt(float(sum) / alphaCs.size());
  } else {
    alphaCMed = 8.;
    alphaCRms = 8.;
  }

  if (alphaFs.size() > 1) {
    alphaFMed = alphaFs[alphaFs.size() / 2 + 1];
    double sum = 0.0;
    for (float alpha : alphaFs)
      sum += std::pow(alpha - alphaFMed, 2);
    alphaFRms = std::sqrt(float(sum) / alphaFs.size());
  } else {
    alphaFMed = 6.;
    alphaFRms = 6.;
  }
  if (debug_)
    dbgPrintf("PUPPI \t alphaC = %+6.2f +- %6.2f (%4lu), alphaF = %+6.2f +- %6.2f (%4lu)\n",
              alphaCMed,
              alphaCRms,
              alphaCs.size(),
              alphaFMed,
              alphaFRms,
              alphaFs.size());
}

void PuppiAlgo::fillPuppi(Region &r) const {
  uint16_t PUPPIW_0p01 = std::round(0.01 * PFParticle::PUPPI_SCALE);
  r.puppi.clear();
  for (PFParticle &p : r.pf) {
    if (p.hwId == l1t::PFCandidate::ChargedHadron || p.hwId == l1t::PFCandidate::Electron ||
        p.hwId == l1t::PFCandidate::Muon) {  // charged
      if (p.hwPuppiWeight > 0) {
        r.puppi.push_back(p);
      }
    } else {  // neutral
      if (p.hwPuppiWeight > PUPPIW_0p01) {
        // FIXME would work better with PUPPI_SCALE being a power of two, to do the shift
        // FIXME done with floats
        int16_t hwPt = (float(p.hwPt) * float(p.hwPuppiWeight) / float(PFParticle::PUPPI_SCALE));
        int16_t hwPtCut = 0, hwAbsEta = r.relativeCoordinates
                                            ? round(r.globalAbsEta(p.floatEta()) * CaloCluster::ETAPHI_SCALE)
                                            : std::abs(p.hwEta);
        for (unsigned int ietaBin = 0, nBins = intPuppiEtaCuts_.size(); ietaBin < nBins; ++ietaBin) {
          if (hwAbsEta < intPuppiEtaCuts_[ietaBin]) {
            hwPtCut = (p.hwId == l1t::PFCandidate::Photon ? intPuppiPtCutsPhotons_[ietaBin] : intPuppiPtCuts_[ietaBin]);
            break;
          }
        }
        if (hwPt > hwPtCut) {
          r.puppi.push_back(p);
          r.puppi.back().hwPt = hwPt;
        }
      }
    }
  }
}
