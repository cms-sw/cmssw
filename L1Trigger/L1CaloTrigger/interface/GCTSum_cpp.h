//------------------------------------
// More logic for Phase2L1GCTSumEmulator
// (based heavily on algo_top.cpp in GCT Sum firmware repo)
//------------------------------------
#ifndef L1Trigger_L1CaloTrigger_GCTSum_cpp
#define L1Trigger_L1CaloTrigger_GCTSum_cpp

#include <ap_int.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "L1Trigger/L1CaloTrigger/interface/GCTSum_h.h"
#include "L1Trigger/L1CaloTrigger/interface/bitonicSort32_GCT_h.h"

namespace p2gctsum {

inline ap_uint<48> pack_gamma_output(const GCTvar& object) {
  ap_uint<48> out = 0;
  if (object.ET == 0)
    return out;
  out = ((ap_uint<48>)object.isBarrel << 47) | ((ap_uint<48>)object.Phi << 19) |
        ((ap_uint<48>)(object.Eta & 0x7f) << 12) | (ap_uint<48>)object.ET;
  // Preserve the high bit of the extended endcap eta in a spare bit without
  // aliasing it into the low bit of phi.
  out[28] = object.Eta[7];
  return out;
}

inline ap_uint<48> pack_had_output(const GCTvar& object) {
  ap_uint<48> out = 0;
  if (object.ET == 0)
    return out;
  out = ((ap_uint<48>)object.isBarrel << 47) | ((ap_uint<48>)object.PtClusterSeed << 27) |
        ((ap_uint<48>)object.Phi << 18) | ((ap_uint<48>)object.Eta << 12) | (ap_uint<48>)object.ET;
  return out;
}

inline void processInputLinks(ap_uint<576> link_in[N_INPUT_LINKS],
                              GCTvar EGs[N_GCT_OBJECTS],
                              GCTvar EGIs[N_GCT_OBJECTS],
                              GCTvar Jets[N_GCT_OBJECTS],
                              GCTvar Taus[N_GCT_OBJECTS],
                              GCTsum& Sums) {
  ap_uint<9> gammaOffset[4] = {0, 0, 120, 240};
  ap_uint<9> hadOffset[4] = {0, 0, 8, 16};

  for (int i = 0; i < N_GCT_CONNECTED; i++) {
    int link_base = 3 * i;
    bool isBarrel = (i > 0);
    ap_uint<9> currentGammaOffset = gammaOffset[i];
    ap_uint<9> currentHadOffset = hadOffset[i];

    ap_uint<576> link_A = link_in[link_base + 0];
    for (int j = 0; j < 12; j++) {
      ap_uint<48> raw_48b = link_A.range(48 * j + 47, 48 * j);
      GCTvar tempObj;
      if (isBarrel) {
        tempObj.getGCTvarBarrelGammas(raw_48b, currentGammaOffset);
      } else {
        tempObj.getGCTvarEndcapGammas(raw_48b, 0);
      }

      if (j < 6)
        EGs[i * 6 + j] = tempObj;
      else
        EGIs[i * 6 + (j - 6)] = tempObj;
    }
    ap_uint<576> link_B = link_in[link_base + 1];
    for (int j = 0; j < 12; j++) {
      ap_uint<48> raw_48b = link_B.range(48 * j + 47, 48 * j);
      GCTvar tempObj;
      tempObj.getGCTvarJetsTaus(raw_48b, currentHadOffset, isBarrel);

      if (j < 6)
        Jets[i * 6 + j] = tempObj;
      else
        Taus[i * 6 + (j - 6)] = tempObj;
    }

    // ABI v2 vector components are already expressed in the global frame.
    Sums.addSource(link_in[link_base + 2]);
  }
}

inline void clear_gctvar(GCTvar& o) {
  o.ET = 0;
  o.Eta = 0;
  o.Phi = 0;
  o.PtClusterSeed = 0;
  o.isBarrel = 0;
}

inline ap_uint<12> saturating_add(ap_uint<12> a, ap_uint<12> b) {
  ap_uint<13> s = (ap_uint<13>)a + (ap_uint<13>)b;
  return (s > 0xFFF) ? (ap_uint<12>)0xFFF : (ap_uint<12>)s;
}

static const int GAMMA_PHI_BINS = 360;
static const int GAMMA_PMUS_PER_REGION = 9;
static const int GAMMA_CRYSTALS_PER_PMU = 5;
static const int GAMMA_REGION_WIDTH = GAMMA_PMUS_PER_REGION * GAMMA_CRYSTALS_PER_PMU;
static const int GAMMA_REGION_OVERLAP = GAMMA_CRYSTALS_PER_PMU;
static const int HAD_PHI_BINS = 24;
static const int HAD_BINS_PER_REGION = 3;
static const int HAD_REGION_OVERLAP = 1;

inline bool is_active_boundary_flag(ap_uint<1> boundary_flag, ap_uint<12> et, ap_uint<1> cleared) {
  return (cleared == 0) && (et != 0) && (boundary_flag == 1);
}

inline void stitch_state_pair_local(ap_uint<12>& ec_et,
                                    ap_uint<1>& ec_cleared,
                                    ap_uint<12>& br_et,
                                    ap_uint<1>& br_cleared) {
  ap_uint<12> sum = saturating_add(ec_et, br_et);

  if (ec_et > br_et) {
    ec_et = sum;
    br_et = 0;
    br_cleared = 1;
  } else {
    br_et = sum;
    ec_et = 0;
    ec_cleared = 1;
  }
}

inline ap_uint<9> normalize_gamma_phi(ap_uint<9> phi) {
  return (phi >= GAMMA_PHI_BINS) ? (ap_uint<9>)(phi - GAMMA_PHI_BINS) : phi;
}

inline ap_uint<5> normalize_had_phi(ap_uint<9> phi) {
  ap_uint<9> p = phi;
  if (p >= 384)
    p -= 384;
  if (p >= 192)
    p -= 192;
  if (p >= 96)
    p -= 96;
  if (p >= 48)
    p -= 48;
  if (p >= HAD_PHI_BINS)
    p -= HAD_PHI_BINS;
  return (ap_uint<5>)p;
}

inline ap_uint<3> gamma_owner_region(ap_uint<9> phi) {
  ap_uint<9> p = normalize_gamma_phi(phi);
  if (p < (1 * GAMMA_REGION_WIDTH))
    return 0;
  if (p < (2 * GAMMA_REGION_WIDTH))
    return 1;
  if (p < (3 * GAMMA_REGION_WIDTH))
    return 2;
  if (p < (4 * GAMMA_REGION_WIDTH))
    return 3;
  if (p < (5 * GAMMA_REGION_WIDTH))
    return 4;
  if (p < (6 * GAMMA_REGION_WIDTH))
    return 5;
  if (p < (7 * GAMMA_REGION_WIDTH))
    return 6;
  return 7;
}

inline ap_uint<3> had_owner_region(ap_uint<9> phi) {
  ap_uint<5> p = normalize_had_phi(phi);
  if (p < (1 * HAD_BINS_PER_REGION))
    return 0;
  if (p < (2 * HAD_BINS_PER_REGION))
    return 1;
  if (p < (3 * HAD_BINS_PER_REGION))
    return 2;
  if (p < (4 * HAD_BINS_PER_REGION))
    return 3;
  if (p < (5 * HAD_BINS_PER_REGION))
    return 4;
  if (p < (6 * HAD_BINS_PER_REGION))
    return 5;
  if (p < (7 * HAD_BINS_PER_REGION))
    return 6;
  return 7;
}

inline bool gamma_phi_in_region_overlap(ap_uint<9> phi, ap_uint<3> region) {
  int start = (int(region) * GAMMA_REGION_WIDTH) - GAMMA_REGION_OVERLAP;
  int end = ((int(region) + 1) * GAMMA_REGION_WIDTH) - 1 + GAMMA_REGION_OVERLAP;
  ap_uint<9> p = normalize_gamma_phi(phi);

  if (start < 0) {
    return (p >= (start + GAMMA_PHI_BINS)) || (p <= end);
  }
  if (end >= GAMMA_PHI_BINS) {
    return (p >= start) || (p <= (end - GAMMA_PHI_BINS));
  }
  return (p >= start) && (p <= end);
}

inline bool had_phi_in_region_overlap(ap_uint<9> phi, ap_uint<3> region) {
  int start = (int(region) * HAD_BINS_PER_REGION) - HAD_REGION_OVERLAP;
  int end = ((int(region) + 1) * HAD_BINS_PER_REGION) - 1 + HAD_REGION_OVERLAP;
  ap_uint<5> p = normalize_had_phi(phi);

  if (start < 0) {
    return (p >= (start + HAD_PHI_BINS)) || (p <= end);
  }
  if (end >= HAD_PHI_BINS) {
    return (p >= start) || (p <= (end - HAD_PHI_BINS));
  }
  return (p >= start) && (p <= end);
}

template <bool IS_GAMMA>
inline void stitch_candidate_pair(const GCTvar& ec,
                                  ap_uint<1> ec_boundary,
                                  ap_uint<3> ec_owner_region,
                                  ap_uint<12>& ec_et,
                                  ap_uint<1>& ec_cleared,
                                  const GCTvar& br,
                                  ap_uint<1> br_boundary,
                                  ap_uint<12>& br_et,
                                  ap_uint<1>& br_cleared) {
  bool in_local_window =
      IS_GAMMA ? gamma_phi_in_region_overlap(br.Phi, ec_owner_region) : had_phi_in_region_overlap(br.Phi, ec_owner_region);

  if (in_local_window && is_active_boundary_flag(ec_boundary, ec_et, ec_cleared) &&
      is_active_boundary_flag(br_boundary, br_et, br_cleared) &&
      (IS_GAMMA ? match_dphi_gamma(ec.Phi, br.Phi) : match_dphi_had(ec.Phi, br.Phi))) {
    stitch_state_pair_local(ec_et, ec_cleared, br_et, br_cleared);
  }
}

template <bool IS_GAMMA>
inline void stitch_boundary_group(GCTvar ec[2], GCTvar br[6], ap_uint<10> endcap_eta, ap_uint<10> barrel_eta) {
  ap_uint<12> ec_et[2];
  ap_uint<12> br_et[6];
  ap_uint<1> ec_cleared[2];
  ap_uint<1> br_cleared[6];
  ap_uint<1> ec_boundary[2];
  ap_uint<1> br_boundary[6];

  for (int i = 0; i < 2; ++i) {
    ec_et[i] = ec[i].ET;
    ec_cleared[i] = 0;
    ec_boundary[i] = (ec[i].isBarrel == 0) && (ec[i].Eta == endcap_eta);
  }

  for (int i = 0; i < 6; ++i) {
    br_et[i] = br[i].ET;
    br_cleared[i] = 0;
    br_boundary[i] = (br[i].isBarrel == 1) && (br[i].Eta == barrel_eta);
  }

  ap_uint<3> ec0_owner_region = IS_GAMMA ? gamma_owner_region(ec[0].Phi) : had_owner_region(ec[0].Phi);
  ap_uint<3> ec1_owner_region = IS_GAMMA ? gamma_owner_region(ec[1].Phi) : had_owner_region(ec[1].Phi);

  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[0],
                                  br_boundary[0], br_et[0], br_cleared[0]);
  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[1],
                                  br_boundary[1], br_et[1], br_cleared[1]);
  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[2],
                                  br_boundary[2], br_et[2], br_cleared[2]);
  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[3],
                                  br_boundary[3], br_et[3], br_cleared[3]);
  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[4],
                                  br_boundary[4], br_et[4], br_cleared[4]);
  stitch_candidate_pair<IS_GAMMA>(ec[0], ec_boundary[0], ec0_owner_region, ec_et[0], ec_cleared[0], br[5],
                                  br_boundary[5], br_et[5], br_cleared[5]);

  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[0],
                                  br_boundary[0], br_et[0], br_cleared[0]);
  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[1],
                                  br_boundary[1], br_et[1], br_cleared[1]);
  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[2],
                                  br_boundary[2], br_et[2], br_cleared[2]);
  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[3],
                                  br_boundary[3], br_et[3], br_cleared[3]);
  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[4],
                                  br_boundary[4], br_et[4], br_cleared[4]);
  stitch_candidate_pair<IS_GAMMA>(ec[1], ec_boundary[1], ec1_owner_region, ec_et[1], ec_cleared[1], br[5],
                                  br_boundary[5], br_et[5], br_cleared[5]);

  for (int i = 0; i < 2; ++i) {
    if (ec_cleared[i] == 1) {
      clear_gctvar(ec[i]);
    } else {
      ec[i].ET = ec_et[i];
    }
  }

  for (int i = 0; i < 6; ++i) {
    if (br_cleared[i] == 1) {
      clear_gctvar(br[i]);
    } else {
      br[i].ET = br_et[i];
    }
  }
}

template <bool IS_GAMMA>
inline void stitch_boundary_collection(GCTvar objects[N_GCT_OBJECTS], ap_uint<10> endcap_eta, ap_uint<10> barrel_eta) {
  for (int barrel_group = 0; barrel_group < 3; ++barrel_group) {
    for (int ec_group = 0; ec_group < 3; ++ec_group) {
      GCTvar ec[2];
      GCTvar br[6];

      int ie_base = ec_group * 2;
      int barrel_base = (barrel_group + 1) * 6;

      for (int i = 0; i < 2; ++i) {
        ec[i] = objects[ie_base + i];
      }

      for (int i = 0; i < 6; ++i) {
        br[i] = objects[barrel_base + i];
      }

      stitch_boundary_group<IS_GAMMA>(ec, br, endcap_eta, barrel_eta);

      for (int i = 0; i < 2; ++i) {
        objects[ie_base + i] = ec[i];
      }

      for (int i = 0; i < 6; ++i) {
        objects[barrel_base + i] = br[i];
      }
    }
  }
}

inline void stitch_gamma_collection(GCTvar objects[N_GCT_OBJECTS]) {
  stitch_boundary_collection<true>(objects, ENDCAP_GAMMA_BOUNDARY_ETA, BARREL_GAMMA_BOUNDARY_ETA);
}

inline void stitch_had_collection(GCTvar objects[N_GCT_OBJECTS]) {
  stitch_boundary_collection<false>(objects, ENDCAP_HAD_BOUNDARY_ETA, BARREL_HAD_BOUNDARY_ETA);
}

inline void updateParams_GCTOutput(GCTvar EGs[N_GCT_OBJECTS],
                                   GCTvar EGIs[N_GCT_OBJECTS],
                                   GCTvar Jets[N_GCT_OBJECTS],
                                   GCTvar Taus[N_GCT_OBJECTS]) {
  stitch_gamma_collection(EGs);
  stitch_gamma_collection(EGIs);
  stitch_had_collection(Jets);
  stitch_had_collection(Taus);
}

inline void sortGCTVars(GCTvar EGsVars[32],
                        GCTvar EGIsVars[32],
                        GCTvar JetsVars[32],
                        GCTvar TausVars[32],
                        GCTvar sortedEGs[32],
                        GCTvar sortedEGIs[32],
                        GCTvar sortedJets[32],
                        GCTvar sortedTaus[32]) {
  bitonicSort32(EGsVars, sortedEGs);
  bitonicSort32(EGIsVars, sortedEGIs);
  bitonicSort32(JetsVars, sortedJets);
  bitonicSort32(TausVars, sortedTaus);
}

inline void combineObjects(GCTvar EGs[24],
                           GCTvar EGIs[24],
                           GCTvar Jets[24],
                           GCTvar Taus[24],
                           GCTvar EGsTop6[6],
                           GCTvar EGIsTop6[6],
                           GCTvar JetsTop6[6],
                           GCTvar TausTop6[6]) {
  GCTvar EGsVars[32];
  GCTvar EGIsVars[32];
  GCTvar JetsVars[32];
  GCTvar TausVars[32];

  GCTvar sortedEGs[32];
  GCTvar sortedEGIs[32];
  GCTvar sortedJets[32];
  GCTvar sortedTaus[32];

  GCTvar dummy;

  for (int i = 0; i < 32; i++) {
    if (i < 24) {
      EGsVars[i] = EGs[i];
      EGIsVars[i] = EGIs[i];
      JetsVars[i] = Jets[i];
      TausVars[i] = Taus[i];
    } else {
      EGsVars[i] = dummy;
      EGIsVars[i] = dummy;
      JetsVars[i] = dummy;
      TausVars[i] = dummy;
    }
  }

  sortGCTVars(EGsVars, EGIsVars, JetsVars, TausVars, sortedEGs, sortedEGIs, sortedJets, sortedTaus);

  for (int i = 0; i < 6; i++) {
    EGsTop6[i] = sortedEGs[31 - i];
    EGIsTop6[i] = sortedEGIs[31 - i];
    JetsTop6[i] = sortedJets[31 - i];
    TausTop6[i] = sortedTaus[31 - i];
  }
}

inline void getHighestInPt(GCTvar EGsTop6[6],
                           GCTvar EGIsTop6[6],
                           GCTvar JetsTop6[6],
                           GCTvar TausTop6[6],
                           GCTvar EGsTop6_reg[6],
                           GCTvar EGIsTop6_reg[6],
                           GCTvar JetsTop6_reg[6],
                           GCTvar TausTop6_reg[6]) {
  for (int i = 0; i < 6; ++i) {
    EGsTop6_reg[i] = EGsTop6[i];
    EGIsTop6_reg[i] = EGIsTop6[i];
    JetsTop6_reg[i] = JetsTop6[i];
    TausTop6_reg[i] = TausTop6[i];
  }
}

inline void processOutLinks(GCTvar EGsTop6[6],
                            GCTvar EGIsTop6[6],
                            GCTvar JetsTop6[6],
                            GCTvar TausTop6[6],
                            const GCTsum& Sums,
                            ap_uint<576> link_out[N_OUTPUT_LINKS]) {
  ap_uint<576> out_link0 = 0;
  ap_uint<576> out_link1 = 0;

  for (int i = 0; i < 6; i++) {
    out_link0.range(i * 48 + 47, i * 48) = pack_gamma_output(EGsTop6[i]);
  }

  for (int i = 0; i < 6; i++) {
    int slot = i + 6;
    out_link0.range(slot * 48 + 47, slot * 48) = pack_gamma_output(EGIsTop6[i]);
  }

  for (int i = 0; i < 6; i++) {
    out_link1.range(i * 48 + 47, i * 48) = pack_had_output(JetsTop6[i]);
  }

  for (int i = 0; i < 6; i++) {
    int slot = i + 6;
    out_link1.range(slot * 48 + 47, slot * 48) = pack_had_output(TausTop6[i]);
  }

  link_out[0] = out_link0;
  link_out[1] = out_link1;
  link_out[2] = Sums.pack();
}

inline void algo_top(ap_uint<576> link_in[N_INPUT_LINKS], ap_uint<576> link_out[N_OUTPUT_LINKS]) {
  GCTvar EGs[24], EGIs[24], Jets[24], Taus[24];
  GCTsum Sums;

  GCTvar EGsTop6[6], EGIsTop6[6], JetsTop6[6], TausTop6[6];
  GCTvar EGsTop6_reg[6], EGIsTop6_reg[6], JetsTop6_reg[6], TausTop6_reg[6];

  processInputLinks(link_in, EGs, EGIs, Jets, Taus, Sums);
  updateParams_GCTOutput(EGs, EGIs, Jets, Taus);
  combineObjects(EGs, EGIs, Jets, Taus, EGsTop6, EGIsTop6, JetsTop6, TausTop6);
  getHighestInPt(EGsTop6, EGIsTop6, JetsTop6, TausTop6, EGsTop6_reg, EGIsTop6_reg, JetsTop6_reg, TausTop6_reg);
  processOutLinks(EGsTop6_reg, EGIsTop6_reg, JetsTop6_reg, TausTop6_reg, Sums, link_out);
}

}  // namespace p2gctsum

#endif
