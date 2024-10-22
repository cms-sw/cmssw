#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace {
  using namespace mkfit;

  [[maybe_unused]] void partitionSeeds0(const TrackerInfo &trk_info,
                                        const TrackVec &in_seeds,
                                        const EventOfHits &eoh,
                                        IterationSeedPartition &part) {
    const size_t size = in_seeds.size();

    for (size_t i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const bool z_dir_pos = S.pz() > 0;

      const auto &hot = S.getLastHitOnTrack();
      const float eta = eoh[hot.layer].refHit(hot.index).eta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      const LayerInfo &outer_brl = trk_info.outer_barrel_layer();

      // Define first (mkFit) layer IDs for each strip subdetector.
      constexpr int tib1_id = 4;
      constexpr int tob1_id = 10;
      constexpr int tecp1_id = 27;
      constexpr int tecn1_id = 54;

      const LayerInfo &tib1 = trk_info.layer(tib1_id);
      const LayerInfo &tob1 = trk_info.layer(tob1_id);

      const LayerInfo &tecp1 = trk_info.layer(tecp1_id);
      const LayerInfo &tecn1 = trk_info.layer(tecn1_id);

      const LayerInfo &tec_first = z_dir_pos ? tecp1 : tecn1;

      const float maxR = S.maxReachRadius();
      float z_at_maxr;

      bool can_reach_outer_brl = S.canReachRadius(outer_brl.rout());
      float z_at_outer_brl;
      bool misses_first_tec;
      if (can_reach_outer_brl) {
        z_at_outer_brl = S.zAtR(outer_brl.rout());
        if (z_dir_pos)
          misses_first_tec = z_at_outer_brl < tec_first.zmin();
        else
          misses_first_tec = z_at_outer_brl > tec_first.zmax();
      } else {
        z_at_maxr = S.zAtR(maxR);
        if (z_dir_pos)
          misses_first_tec = z_at_maxr < tec_first.zmin();
        else
          misses_first_tec = z_at_maxr > tec_first.zmax();
      }

      if (misses_first_tec) {
        reg = TrackerInfo::Reg_Barrel;
      } else {
        if ((S.canReachRadius(tib1.rin()) && tib1.is_within_z_limits(S.zAtR(tib1.rin()))) ||
            (S.canReachRadius(tob1.rin()) && tob1.is_within_z_limits(S.zAtR(tob1.rin())))) {
          reg = z_dir_pos ? TrackerInfo::Reg_Transition_Pos : TrackerInfo::Reg_Transition_Neg;
        } else {
          reg = z_dir_pos ? TrackerInfo::Reg_Endcap_Pos : TrackerInfo::Reg_Endcap_Neg;
        }
      }

      part.m_region[i] = reg;
      if (part.m_phi_eta_foo)
        part.m_phi_eta_foo(eoh[hot.layer].refHit(hot.index).phi(), eta);
    }
  }

  [[maybe_unused]] void partitionSeeds1(const TrackerInfo &trk_info,
                                        const TrackVec &in_seeds,
                                        const EventOfHits &eoh,
                                        IterationSeedPartition &part) {
    // Define first (mkFit) layer IDs for each strip subdetector.
    constexpr int tib1_id = 4;
    constexpr int tob1_id = 10;
    constexpr int tidp1_id = 21;
    constexpr int tidn1_id = 48;
    constexpr int tecp1_id = 27;
    constexpr int tecn1_id = 54;

    const LayerInfo &tib1 = trk_info.layer(tib1_id);
    const LayerInfo &tob1 = trk_info.layer(tob1_id);

    const LayerInfo &tidp1 = trk_info.layer(tidp1_id);
    const LayerInfo &tidn1 = trk_info.layer(tidn1_id);

    const LayerInfo &tecp1 = trk_info.layer(tecp1_id);
    const LayerInfo &tecn1 = trk_info.layer(tecn1_id);

    // Merge first two layers to account for mono/stereo coverage.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info;
    const float tidp_rin = std::min(L[tidp1_id].rin(), L[tidp1_id + 1].rin());
    const float tidp_rout = std::max(L[tidp1_id].rout(), L[tidp1_id + 1].rout());
    const float tecp_rin = std::min(L[tecp1_id].rin(), L[tecp1_id + 1].rin());
    const float tecp_rout = std::max(L[tecp1_id].rout(), L[tecp1_id + 1].rout());
    const float tidn_rin = std::min(L[tidn1_id].rin(), L[tidn1_id + 1].rin());
    const float tidn_rout = std::max(L[tidn1_id].rout(), L[tidn1_id + 1].rout());
    const float tecn_rin = std::min(L[tecn1_id].rin(), L[tecn1_id + 1].rin());
    const float tecn_rout = std::max(L[tecn1_id].rout(), L[tecn1_id + 1].rout());

    // Bias towards more aggressive transition-region assignemnts.
    // With current tunning it seems to make things a bit worse.
    const float tid_z_extra = 0.0f;  // 5.0f;
    const float tec_z_extra = 0.0f;  // 10.0f;

    const size_t size = in_seeds.size();

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

    for (size_t i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const auto &hot = S.getLastHitOnTrack();
      const float eta = eoh[hot.layer].refHit(hot.index).eta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      const bool z_dir_pos = S.pz() > 0;
      const float maxR = S.maxReachRadius();

      if (z_dir_pos) {
        const bool in_tib = barrel_pos_check(S, maxR, tib1.rin(), tib1.zmax());
        const bool in_tob = barrel_pos_check(S, maxR, tob1.rin(), tob1.zmax());

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Pos;
        } else {
          const bool in_tid = endcap_pos_check(S, maxR, tidp_rout, tidp_rin, tidp1.zmin() - tid_z_extra);
          const bool in_tec = endcap_pos_check(S, maxR, tecp_rout, tecp_rin, tecp1.zmin() - tec_z_extra);

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
          }
        }
      } else {
        const bool in_tib = barrel_neg_check(S, maxR, tib1.rin(), tib1.zmin());
        const bool in_tob = barrel_neg_check(S, maxR, tob1.rin(), tob1.zmin());

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Neg;
        } else {
          const bool in_tid = endcap_neg_check(S, maxR, tidn_rout, tidn_rin, tidn1.zmax() + tid_z_extra);
          const bool in_tec = endcap_neg_check(S, maxR, tecn_rout, tecn_rin, tecn1.zmax() + tec_z_extra);

          if (!in_tid && !in_tec) {
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
    // Define first (mkFit) layer IDs for each strip subdetector.
    constexpr int tib1_id = 4;
    constexpr int tob1_id = 10;
    constexpr int tidp1_id = 21;
    constexpr int tidn1_id = 48;
    constexpr int tecp1_id = 27;
    constexpr int tecn1_id = 54;

    const LayerInfo &tib1 = trk_info.layer(tib1_id);
    const LayerInfo &tob1 = trk_info.layer(tob1_id);

    const LayerInfo &tidp1 = trk_info.layer(tidp1_id);
    const LayerInfo &tidn1 = trk_info.layer(tidn1_id);

    const LayerInfo &tecp1 = trk_info.layer(tecp1_id);
    const LayerInfo &tecn1 = trk_info.layer(tecn1_id);

    // Merge first two layers to account for mono/stereo coverage.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info;
    const float tidp_rin = std::min(L[tidp1_id].rin(), L[tidp1_id + 1].rin());
    const float tidp_rout = std::max(L[tidp1_id].rout(), L[tidp1_id + 1].rout());
    const float tecp_rin = std::min(L[tecp1_id].rin(), L[tecp1_id + 1].rin());
    const float tecp_rout = std::max(L[tecp1_id].rout(), L[tecp1_id + 1].rout());
    const float tidn_rin = std::min(L[tidn1_id].rin(), L[tidn1_id + 1].rin());
    const float tidn_rout = std::max(L[tidn1_id].rout(), L[tidn1_id + 1].rout());
    const float tecn_rin = std::min(L[tecn1_id].rin(), L[tecn1_id + 1].rin());
    const float tecn_rout = std::max(L[tecn1_id].rout(), L[tecn1_id + 1].rout());

    // Bias towards more aggressive transition-region assignemnts.
    // With current tunning it seems to make things a bit worse.
    const float tid_z_extra = 0.0f;  //  5.0f;
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
        bool in_tib = barrel_pos_check(S, maxR, tib1.rin(), tib1.zmax(), "TIBp");
        bool in_tob = barrel_pos_check(S, maxR, tob1.rin(), tob1.zmax(), "TOBp");

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Pos;
          printf("  --> region = %d, endcap pos\n", reg);
        } else {
          bool in_tid = endcap_pos_check(S, maxR, tidp_rout, tidp_rin, tidp1.zmin() - tid_z_extra, "TIDp");
          bool in_tec = endcap_pos_check(S, maxR, tecp_rout, tecp_rin, tecp1.zmin() - tec_z_extra, "TECp");

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
            printf("  --> region = %d, barrel\n", reg);
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
            printf("  --> region = %d, transition pos\n", reg);
          }
        }
      } else {
        bool in_tib = barrel_neg_check(S, maxR, tib1.rin(), tib1.zmin(), "TIBn");
        bool in_tob = barrel_neg_check(S, maxR, tob1.rin(), tob1.zmin(), "TOBn");

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Neg;
          printf("  --> region = %d, endcap neg\n", reg);
        } else {
          bool in_tid = endcap_neg_check(S, maxR, tidn_rout, tidn_rin, tidn1.zmax() + tid_z_extra, "TIDn");
          bool in_tec = endcap_neg_check(S, maxR, tecn_rout, tecn_rin, tecn1.zmax() + tec_z_extra, "TECn");

          if (!in_tid && !in_tec) {
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
      IterationConfig::register_seed_partitioner("phase1:0", partitionSeeds0);
      IterationConfig::register_seed_partitioner("phase1:1", partitionSeeds1);
      IterationConfig::register_seed_partitioner("phase1:1:debug", partitionSeeds1debug);
    }
  } rsp_instance;
}  // namespace
