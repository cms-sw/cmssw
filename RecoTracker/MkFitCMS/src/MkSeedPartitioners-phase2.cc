#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace {
  using namespace mkfit;

  // named constants for useful layers (l/u for lower/upper)
  constexpr int tecp1l_id = 28;
  constexpr int tecp1u_id = 29;
  constexpr int tecp2l_id = 30;
  constexpr int tecp2u_id = 31;
  constexpr int tecn1l_id = 50;
  constexpr int tecn1u_id = 51;
  constexpr int tecn2l_id = 52;
  constexpr int tecntu_id = 53;

  [[maybe_unused]] void partitionSeeds1(const TrackerInfo &trk_info,
                                        const TrackVec &in_seeds,
                                        const EventOfHits &eoh,
                                        IterationSeedPartition &part) {
    // Seeds are placed into eta regions and sorted on region + eta.

    // Merge mono and stereo limits for relevant layers / parameters.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info;
    const float tecp1_rin = std::min(L[tecp1l_id].rin(), L[tecp1u_id].rin());
    const float tecp1_rout = std::max(L[tecp1l_id].rout(), L[tecp1u_id].rout());
    const float tecp1_zmin = std::min(L[tecp1l_id].zmin(), L[tecp1u_id].zmin());

    const float tecp2_rin = std::min(L[tecp2l_id].rin(), L[tecp2u_id].rin());
    const float tecp2_zmax = std::max(L[tecp2l_id].zmax(), L[tecp2u_id].zmax());

    const float tecn1_rin = std::min(L[tecn1l_id].rin(), L[tecn1u_id].rin());
    const float tecn1_rout = std::max(L[tecn1l_id].rout(), L[tecn1u_id].rout());
    const float tecn1_zmax = std::max(L[tecn1l_id].zmax(), L[tecn1u_id].zmax());

    const float tecn2_rin = std::min(L[tecn2l_id].rin(), L[tecntu_id].rin());
    const float tecn2_zmin = std::min(L[tecn2l_id].zmin(), L[tecntu_id].zmin());

    const float tec_z_extra = 0.0f;  // 10.0f;

    const int size = in_seeds.size();

    auto barrel_pos_check = [](const Track &S, float maxR, float rin, float zmax) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) < zmax;
      return inside;
    };

    auto barrel_neg_check = [](const Track &S, float maxR, float rin, float zmin) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) > zmin;
      return inside;
    };

    auto endcap_pos_check = [](const Track &S, float maxR, float rout, float rin, float zmin) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) > zmin : (maxR > rin && S.zAtR(maxR) > zmin);
      return inside;
    };

    auto endcap_neg_check = [](const Track &S, float maxR, float rout, float rin, float zmax) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) < zmax : (maxR > rin && S.zAtR(maxR) < zmax);
      return inside;
    };

    for (int i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      HitOnTrack hot = S.getLastHitOnTrack();
      float eta = eoh[hot.layer].refHit(hot.index).eta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      const bool z_dir_pos = S.pz() > 0;
      const float maxR = S.maxReachRadius();

      if (z_dir_pos) {
        bool in_tec_as_brl = barrel_pos_check(S, maxR, tecp2_rin, tecp2_zmax);

        if (!in_tec_as_brl) {
          reg = TrackerInfo::Reg_Endcap_Pos;
        } else {
          bool in_tec = endcap_pos_check(S, maxR, tecp1_rout, tecp1_rin, tecp1_zmin - tec_z_extra);

          if (!in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
          }
        }
      } else {
        bool in_tec_as_brl = barrel_neg_check(S, maxR, tecn2_rin, tecn2_zmin);

        if (!in_tec_as_brl) {
          reg = TrackerInfo::Reg_Endcap_Neg;
        } else {
          bool in_tec = endcap_neg_check(S, maxR, tecn1_rout, tecn1_rin, tecn1_zmax + tec_z_extra);

          if (!in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Neg;
          }
        }
      }

      part.m_region[i] = reg;
      if (part.m_phi_eta_foo)
        part.m_phi_eta_foo(eoh[hot.layer].refHit(hot.index).phi(), eta);
    }
  }

  [[maybe_unused]] void partitionSeeds1debug(const TrackerInfo &trk_info,
                                             const TrackVec &in_seeds,
                                             const EventOfHits &eoh,
                                             IterationSeedPartition &part) {
    // Seeds are placed into eta regions and sorted on region + eta.

    // Merge mono and stereo limits for relevant layers / parameters.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info;
    const float tecp1_rin = std::min(L[tecp1l_id].rin(), L[tecp1u_id].rin());
    const float tecp1_rout = std::max(L[tecp1l_id].rout(), L[tecp1u_id].rout());
    const float tecp1_zmin = std::min(L[tecp1l_id].zmin(), L[tecp1u_id].zmin());

    const float tecp2_rin = std::min(L[tecp2l_id].rin(), L[tecp2u_id].rin());
    const float tecp2_zmax = std::max(L[tecp2l_id].zmax(), L[tecp2u_id].zmax());

    const float tecn1_rin = std::min(L[tecn1l_id].rin(), L[tecn1u_id].rin());
    const float tecn1_rout = std::max(L[tecn1l_id].rout(), L[tecn1u_id].rout());
    const float tecn1_zmax = std::max(L[tecn1l_id].zmax(), L[tecn1u_id].zmax());

    const float tecn2_rin = std::min(L[tecn2l_id].rin(), L[tecntu_id].rin());
    const float tecn2_zmin = std::min(L[tecn2l_id].zmin(), L[tecntu_id].zmin());

    const float tec_z_extra = 0.0f;  // 10.0f;

    const int size = in_seeds.size();

    auto barrel_pos_check = [](const Track &S, float maxR, float rin, float zmax, const char *det) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) < zmax;

      printf("  in_%s=%d  maxR=%7.3f, rin=%7.3f -- ", det, inside, maxR, rin);
      if (maxR > rin) {
        printf("maxR > rin:   S.zAtR(rin) < zmax  -- %.3f <? %.3f\n", S.zAtR(rin), zmax);
      } else {
        printf("maxR < rin: no pie.\n");
      }

      return inside;
    };

    auto barrel_neg_check = [](const Track &S, float maxR, float rin, float zmin, const char *det) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) > zmin;

      printf("  in_%s=%d  maxR=%7.3f, rin=%7.3f -- ", det, inside, maxR, rin);
      if (maxR > rin) {
        printf("maxR > rin:   S.zAtR(rin) > zmin  -- %.3f >? %.3f\n", S.zAtR(rin), zmin);
      } else {
        printf("maxR < rin: no pie.\n");
      }

      return inside;
    };

    auto endcap_pos_check = [](const Track &S, float maxR, float rout, float rin, float zmin, const char *det) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) > zmin : (maxR > rin && S.zAtR(maxR) > zmin);

      printf("  in_%s=%d  maxR=%7.3f, rout=%7.3f, rin=%7.3f -- ", det, inside, maxR, rout, rin);
      if (maxR > rout) {
        printf("maxR > rout:  S.zAtR(rout) > zmin  -- %.3f >? %.3f\n", S.zAtR(rout), zmin);
      } else if (maxR > rin) {
        printf("maxR > rin:   S.zAtR(maxR) > zmin) -- %.3f >? %.3f\n", S.zAtR(maxR), zmin);
      } else {
        printf("maxR < rin: no pie.\n");
      }

      return inside;
    };

    auto endcap_neg_check = [](const Track &S, float maxR, float rout, float rin, float zmax, const char *det) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) < zmax : (maxR > rin && S.zAtR(maxR) < zmax);

      printf("  in_%s=%d  maxR=%7.3f, rout=%7.3f, rin=%7.3f -- ", det, inside, maxR, rout, rin);
      if (maxR > rout) {
        printf("maxR > rout:  S.zAtR(rout) < zmax  -- %.3f <? %.3f\n", S.zAtR(rout), zmax);
      } else if (maxR > rin) {
        printf("maxR > rin:   S.zAtR(maxR) < zmax  -- %.3f <? %.3f\n", S.zAtR(maxR), zmax);
      } else {
        printf("maxR < rin: no pie.\n");
      }

      return inside;
    };

    for (int i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      HitOnTrack hot = S.getLastHitOnTrack();
      float eta = eoh[hot.layer].refHit(hot.index).eta();
      // float  eta = S.momEta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      const bool z_dir_pos = S.pz() > 0;
      const float maxR = S.maxReachRadius();

      printf("partitionSeeds1debug seed index %d, z_dir_pos=%d (pz=%.3f), maxR=%.3f\n", i, z_dir_pos, S.pz(), maxR);

      if (z_dir_pos) {
        bool in_tec_as_brl = barrel_pos_check(S, maxR, tecp2_rin, tecp2_zmax, "TECasBarrelp");

        if (!in_tec_as_brl) {
          reg = TrackerInfo::Reg_Endcap_Pos;
          printf("  --> region = %d, endcap pos\n", reg);
        } else {
          bool in_tec = endcap_pos_check(S, maxR, tecp1_rout, tecp1_rin, tecp1_zmin - tec_z_extra, "TECp");

          if (!in_tec) {
            reg = TrackerInfo::Reg_Barrel;
            printf("  --> region = %d, barrel\n", reg);
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
            printf("  --> region = %d, transition pos\n", reg);
          }
        }
      } else {
        bool in_tec_as_brl = barrel_neg_check(S, maxR, tecn2_rin, tecn2_zmin, "TECasBarreln");

        if (!in_tec_as_brl) {
          reg = TrackerInfo::Reg_Endcap_Neg;
          printf("  --> region = %d, endcap neg\n", reg);
        } else {
          bool in_tec = endcap_neg_check(S, maxR, tecn1_rout, tecn1_rin, tecn1_zmax + tec_z_extra, "TECn");

          if (!in_tec) {
            reg = TrackerInfo::Reg_Barrel;
            printf("  --> region = %d, barrel\n", reg);
          } else {
            reg = TrackerInfo::Reg_Transition_Neg;
            printf("  --> region = %d, transition neg\n", reg);
          }
        }
      }

      part.m_region[i] = reg;
      if (part.m_phi_eta_foo)
        part.m_phi_eta_foo(eoh[hot.layer].refHit(hot.index).phi(), eta);
    }
  }

  CMS_SA_ALLOW struct register_seed_partitioners {
    register_seed_partitioners() {
      IterationConfig::register_seed_partitioner("phase2:1", partitionSeeds1);
      IterationConfig::register_seed_partitioner("phase2:1:debug", partitionSeeds1debug);
    }
  } rsp_instance;
}  // namespace
