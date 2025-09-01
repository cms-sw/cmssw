#ifndef L1Trigger_Phase2L1ParticleFlow_JUMP_h
#define L1Trigger_Phase2L1ParticleFlow_JUMP_h

#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1PFMetEmulator.h"

#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "nlohmann/json.hpp"

#include <vector>
#include <numeric>
#include <array>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "ap_int.h"
#include "ap_fixed.h"

namespace L1JUMPEmu {
  /*
    Emulator for the JUMP Algorithm
    DPS Note publicly available on CDS: CMS DP-2025/023
    JUMP: Jet Uncertainty-aware MET Prediction
    - Approximate L1 Jet energy resolution by pT, eta value
    - Apply the estimated resolution to MET
  */
  struct JER_param {
    std::array<ap_fixed<11, 1>, 5> par0;   // eta.par0 (slope)
    std::array<ap_fixed<8, 5>, 5> par1;    // eta.par1 (offset)
    std::array<L1METEmu::eta_t, 4> edges;  // |eta| boundaries in HW scale
  };

  inline const JER_param& Get_jer_param() {
    static JER_param P = []() {
      JER_param t{};
      edm::FileInPath fip("L1Trigger/Phase2L1ParticleFlow/data/met/l1jump_jer_v1.json");
      std::ifstream in(fip.fullPath());
      if (!in)
        throw cms::Exception("FileNotFound") << fip.fullPath();
      nlohmann::json j;
      in >> j;

      for (int i = 0; i < 5; ++i) {
        t.par0[i] = ap_fixed<11, 1>(j["eta"]["par0"][i].get<double>());
        t.par1[i] = ap_fixed<8, 5>(j["eta"]["par1"][i].get<double>());
      }
      for (int i = 0; i < 4; ++i) {
        t.edges[i] = l1ct::Scales::makeGlbEta(j["eta_edges"][i].get<double>());
      }
      return t;
    }();
    return P;
  }

  inline void Get_dPt(const l1ct::Jet jet, L1METEmu::proj2_t& dPx_2, L1METEmu::proj2_t& dPy_2) {
    /*
      L1 Jet Energy Resolution parameterization
      - Fitted σ(pT)/pT as a function of jet pT in each η region (detector boundary at η≈1.3, 1.7, 2.5, 3.0)
      - Derived from simulated QCD multijet samples to calculate detector‐dependent resolution
      - σ(pT) ≈ eta_par0[i] * pT + eta_par1[i]
    */

    const auto& J = Get_jer_param();

    L1METEmu::eta_t abseta = abs(jet.hwEta.to_float());
    int etabin = 0;
    for (uint i = 0; i < 4;) {
      if (abseta < J.edges[i]) {
        etabin = i+1;
        break;
      }
    }

    dPx_2 = 0;
    dPy_2 = 0;
    l1ct::Sum jet_resolution;
    jet_resolution.hwPt = J.par0[etabin] * jet.hwPt + J.par1[etabin];
    jet_resolution.hwPhi = jet.hwPhi;
    L1METEmu::Particle_xy dpt_xy = L1METEmu::Get_xy(jet_resolution.hwPt, jet_resolution.hwPhi);

    dPx_2 = dpt_xy.hwPx * dpt_xy.hwPx;
    dPy_2 = dpt_xy.hwPy * dpt_xy.hwPy;
  }

  inline void Met_dPt(const std::vector<l1ct::Jet> jets, L1METEmu::proj2_t& dPx_2, L1METEmu::proj2_t& dPy_2) {
    L1METEmu::proj2_t each_dPx2 = 0;
    L1METEmu::proj2_t each_dPy2 = 0;

    L1METEmu::proj2_t sum_dPx2 = 0;
    L1METEmu::proj2_t sum_dPy2 = 0;

    for (uint i = 0; i < jets.size(); i++) {
      Get_dPt(jets[i], each_dPx2, each_dPy2);
      sum_dPx2 += each_dPx2;
      sum_dPy2 += each_dPy2;
    }

    dPx_2 = sum_dPx2;
    dPy_2 = sum_dPy2;
  }
}  // namespace L1JUMPEmu

inline void JUMP_emu(const l1ct::Sum inMet, const std::vector<l1ct::Jet> jets, l1ct::Sum& outMet) {
  L1METEmu::Particle_xy inMet_xy = L1METEmu::Get_xy(inMet.hwPt, inMet.hwPhi);

  L1METEmu::proj2_t dPx_2;
  L1METEmu::proj2_t dPy_2;
  L1JUMPEmu::Met_dPt(jets, dPx_2, dPy_2);

  L1METEmu::Particle_xy outMet_xy;
  outMet_xy.hwPx = (inMet_xy.hwPx > 0) ? inMet_xy.hwPx + L1METEmu::proj2_t(sqrt(dPx_2.to_float()))
                                       : inMet_xy.hwPx - L1METEmu::proj2_t(sqrt(dPx_2.to_float()));
  outMet_xy.hwPy = (inMet_xy.hwPy > 0) ? inMet_xy.hwPy + L1METEmu::proj2_t(sqrt(dPy_2.to_float()))
                                       : inMet_xy.hwPy - L1METEmu::proj2_t(sqrt(dPy_2.to_float()));
  L1METEmu::pxpy_to_ptphi(outMet_xy, outMet);
}

#endif
