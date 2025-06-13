#ifndef L1Trigger_Phase2L1ParticleFlow_JUMP_h
#define L1Trigger_Phase2L1ParticleFlow_JUMP_h

#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1PFMetEmulator.h"

#include <vector>
#include <numeric>
#include <algorithm>
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

  inline void Get_dPt(l1ct::Jet jet, L1METEmu::proj2_t& dPx_2, L1METEmu::proj2_t& dPy_2) {
    /*
      L1 Jet Energy Resolution parameterization
      - Fitted σ(pT)/pT as a function of jet pT in each η region (detector boundary at η≈1.3, 1.7, 2.5, 3.0)
      - Derived from simulated QCD multijet samples to calculate detector‐dependent resolution
      - σ(pT) ≈ eta_par1[i] * pT + eta_par2[i]
    */

    ap_fixed<11, 1> eta_par1[5] = {0, 0.073, 0.247, 0.128, 0.091};
    ap_fixed<8, 5> eta_par2[5] = {0, 12.322, 6.061, 10.944, 12.660};

    L1METEmu::eta_t eta_edges[4];
    float eta_boundaries[4] = {1.3, 1.7, 2.5, 3.0};
    for (uint i=0; i < 4; i++){
      eta_edges[i] = l1ct::Scales::makeGlbEta(eta_boundaries[i]);
    }

    L1METEmu::eta_t abseta = abs(jet.hwEta.to_float());
    int etabin = 0;
    if (abseta == 0.00)
      etabin = 1;
    else if (abseta < eta_edges[0])
      etabin = 1;
    else if (abseta < eta_edges[1])
      etabin = 2;
    else if (abseta < eta_edges[2])
      etabin = 3;
    else if (abseta < eta_edges[3])
      etabin = 4;
    else
      etabin = 0;

    dPx_2 = 0;
    dPy_2 = 0;
    l1ct::Sum jet_resolution;
    jet_resolution.hwPt = eta_par1[etabin] * jet.hwPt + eta_par2[etabin];
    jet_resolution.hwPhi = jet.hwPhi;
    L1METEmu::Particle_xy dpt_xy = L1METEmu::Get_xy(jet_resolution.hwPt, jet_resolution.hwPhi);

    dPx_2 = dpt_xy.hwPx * dpt_xy.hwPx;
    dPy_2 = dpt_xy.hwPy * dpt_xy.hwPy;
    
  }

  inline void Met_dPt(std::vector<l1ct::Jet> jets, L1METEmu::proj2_t& dPx_2, L1METEmu::proj2_t& dPy_2) {
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

inline void JUMP_emu(l1ct::Sum inMet, std::vector<l1ct::Jet> jets, l1ct::Sum& outMet) {

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

  return;
}

#endif
