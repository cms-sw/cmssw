#include "L1Trigger/Phase2L1ParticleFlow/interface/LinearizedPuppiAlgo.h"
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

LinearizedPuppiAlgo::LinearizedPuppiAlgo(const edm::ParameterSet &iConfig)
    : PuppiAlgo(iConfig),
      puppiPriors_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPriors"))),
      puppiPriorsPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPriorsPhotons"))),
      puppiPtSlopes_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtSlopes"))),
      puppiPtSlopesPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtSlopesPhotons"))),
      puppiPtZeros_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtZeros"))),
      puppiPtZerosPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtZerosPhotons"))),
      puppiAlphaSlopes_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaSlopes"))),
      puppiAlphaSlopesPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaSlopesPhotons"))),
      puppiAlphaZeros_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaZeros"))),
      puppiAlphaZerosPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaZerosPhotons"))),
      puppiAlphaCrops_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaCrops"))),
      puppiAlphaCropsPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiAlphaCropsPhotons"))) {
  if (puppiPriors_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPriors\n");
  if (puppiPtSlopes_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPtSlopes\n");
  if (puppiPtZeros_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPtZeros\n");
  if (puppiAlphaSlopes_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaSlopes\n");
  if (puppiAlphaZeros_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaZeros\n");
  if (puppiAlphaCrops_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaCrops\n");
  if (puppiPriorsPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPriorsPhotons\n");
  if (puppiPtSlopesPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPtSlopesPhotons\n");
  if (puppiPtZerosPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiPtZerosPhotons\n");
  if (puppiAlphaSlopesPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaSlopesPhotons\n");
  if (puppiAlphaZerosPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaZerosPhotons\n");
  if (puppiAlphaCropsPhotons_.size() != puppiEtaCuts_.size())
    throw cms::Exception("Configuration", "Mismatched lenght for puppiAlphaCropsPhotons\n");
}

LinearizedPuppiAlgo::~LinearizedPuppiAlgo() {}

const std::vector<std::string> &LinearizedPuppiAlgo::puGlobalNames() const {
  static const std::vector<std::string> names_{};
  return names_;
}
void LinearizedPuppiAlgo::doPUGlobals(const std::vector<Region> &rs,
                                      float z0,
                                      float npu,
                                      std::vector<float> &globals) const {
  globals.clear();
}
void LinearizedPuppiAlgo::runNeutralsPU(Region &r, float z0, float npu, const std::vector<float> &globals) const {
  std::vector<float> alphaC, alphaF;
  PuppiAlgo::computePuppiAlphas(r, alphaC, alphaF);
  computePuppiWeights(r, npu, alphaC, alphaF);
  PuppiAlgo::fillPuppi(r);
}

void LinearizedPuppiAlgo::computePuppiWeights(Region &r,
                                              float npu,
                                              const std::vector<float> &alphaC,
                                              const std::vector<float> &alphaF) const {
  if (debug_ && npu > 0)
    dbgPrintf("LinPup\t npu estimate %7.2f --> log(npu/200) = %+6.2f \n", npu, std::log(npu / 200.f));
  for (unsigned int ip = 0, np = r.pf.size(); ip < np; ++ip) {
    PFParticle &p = r.pf[ip];
    // charged
    if (p.hwId == l1t::PFCandidate::ChargedHadron || p.hwId == l1t::PFCandidate::Electron ||
        p.hwId == l1t::PFCandidate::Muon) {
      p.setPuppiW(p.chargedPV || p.hwId == l1t::PFCandidate::Muon ? 1.0 : 0);
      if (debug_ == 2)
        dbgPrintf(
            "LinPup\t charged id %1d pt %7.2f eta %+5.2f phi %+5.2f   fromPV %1d                                       "
            "                        --> puppi weight %.3f   puppi pt %7.2f \n",
            p.hwId,
            p.floatPt(),
            p.floatEta(),
            p.floatPhi(),
            p.chargedPV,
            p.floatPuppiW(),
            p.floatPt() * p.floatPuppiW());
      continue;
    }
    // neutral
    float absEta = r.relativeCoordinates ? r.globalAbsEta(p.floatEta()) : std::abs(p.floatEta());
    bool central = absEta < etaCharged_;  // FIXME could make a better integer implementation
    bool photon = (p.hwId == l1t::PFCandidate::Photon);
    // get alpha
    float alpha = central ? alphaC[ip] : alphaF[ip];
    alpha = (alpha > 0 ? std::log(alpha) : 0);
    // get eta bin
    unsigned int ietaBin = 0, lastBin = puppiEtaCuts_.size() - 1;
    while (ietaBin < lastBin && absEta > puppiEtaCuts_[ietaBin]) {
      ietaBin++;
    }
    float alphaZero = (photon ? puppiAlphaZerosPhotons_ : puppiAlphaZeros_)[ietaBin];
    float alphaSlope = (photon ? puppiAlphaSlopesPhotons_ : puppiAlphaSlopes_)[ietaBin];
    float alphaCrop = (photon ? puppiAlphaCropsPhotons_ : puppiAlphaCrops_)[ietaBin];
    float x2a = std::clamp(alphaSlope * (alpha - alphaZero), -alphaCrop, alphaCrop);
    // weight by pT
    float ptZero = (photon ? puppiPtZerosPhotons_ : puppiPtZeros_)[ietaBin];
    float ptSlope = (photon ? puppiPtSlopesPhotons_ : puppiPtSlopes_)[ietaBin];
    float x2pt = ptSlope * (p.floatPt() - ptZero);
    // weight by prior
    float prior = (photon ? puppiPriorsPhotons_ : puppiPriors_)[ietaBin];
    float x2prior = (npu > 0 ? std::log(npu / 200.f) : 0) + prior;
    // total
    float x2 = x2a + x2pt - x2prior;
    p.setPuppiW(1.0 / (1.0 + std::exp(-x2)));
    if (debug_ == 1 || debug_ == 2 || debug_ == int(10 + ietaBin))
      dbgPrintf(
          "LinPup\t neutral id %1d pt %7.2f eta %+5.2f phi %+5.2f   alpha %+6.2f   x2a %+5.2f   x2pt %+6.2f   x2prior "
          "%+6.2f -->  x2 %+6.2f --> puppi weight %.3f   puppi pt %7.2f \n",
          p.hwId,
          p.floatPt(),
          p.floatEta(),
          p.floatPhi(),
          alpha,
          x2a,
          x2pt,
          -x2prior,
          x2,
          p.floatPuppiW(),
          p.floatPt() * p.floatPuppiW());
  }
}
