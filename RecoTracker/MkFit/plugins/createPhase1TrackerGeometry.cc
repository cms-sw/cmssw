//-------------------
// Phase1 tracker geometry
//-------------------

#include "Config.h"
#include "Debug.h"
#include "TrackerInfo.h"
#include "mkFit/IterationConfig.h"
#include "mkFit/HitStructures.h"

#include <functional>

using namespace mkfit;

namespace {
#include "createPhase1TrackerGeometryAutoGen.acc"

  void setupSteeringParamsIter0(IterationConfig &ic) {
    ic.m_region_order[0] = TrackerInfo::Reg_Transition_Pos;
    ic.m_region_order[1] = TrackerInfo::Reg_Transition_Neg;
    ic.m_region_order[2] = TrackerInfo::Reg_Endcap_Pos;
    ic.m_region_order[3] = TrackerInfo::Reg_Endcap_Neg;
    ic.m_region_order[4] = TrackerInfo::Reg_Barrel;

    {
      SteeringParams &sp = ic.m_steering_params[TrackerInfo::Reg_Endcap_Neg];
      sp.reserve_plan(3 + 3 + 6 + 18);  // BPix + FPix- + TID- + TEC-; BPix4 is out of acceptance
      sp.fill_plan(0, 1, false, true);  // bk-fit only
      sp.append_plan(2, true);          // pick-up only
      sp.append_plan(45, false);
      sp.append_plan(46, false);
      sp.append_plan(47, false);
      sp.fill_plan(48, 53);  // TID,  6 disks (3 mono + 3 stereo)
      sp.fill_plan(54, 71);  // TEC, 18 disks (3 mono + 3 stereo)
      sp.finalize_plan();
    }
    {
      SteeringParams &sp = ic.m_steering_params[TrackerInfo::Reg_Transition_Neg];
      sp.reserve_plan(4 + 3 + 6 + 6 + 8 + 18);  // BPix + FPix- + TIB + TID- + TOB + TEC-
      sp.fill_plan(0, 1, false, true);          // bk-fit only
      sp.append_plan(2, true);
      sp.append_plan(3, false);
      sp.append_plan(45, false);
      sp.append_plan(46, false);
      sp.append_plan(47, false);
      sp.fill_plan(4, 9);    // TIB,  6 layers (4 mono + 2 stereo)
      sp.fill_plan(48, 53);  // TID,  6 disks  (3 mono + 3 stereo)
      sp.fill_plan(10, 17);  // TOB,  8 layers (6 mono + 2 stereo)
      sp.fill_plan(54, 71);  // TEC, 18 disks  (9 mono + 9 stereo)
      sp.finalize_plan();
    }
    {
      SteeringParams &sp = ic.m_steering_params[TrackerInfo::Reg_Barrel];
      sp.reserve_plan(4 + 6 + 8);       // BPix + TIB + TOB
      sp.fill_plan(0, 1, false, true);  // bk-fit only
      sp.append_plan(2, true);          // pickup-only
      sp.append_plan(3, false);
      sp.fill_plan(4, 9);    // TIB, 6 layers (4 mono + 2 stereo)
      sp.fill_plan(10, 17);  // TOB, 8 layers (6 mono + 2 stereo)
      sp.finalize_plan();
    }
    {
      SteeringParams &sp = ic.m_steering_params[TrackerInfo::Reg_Transition_Pos];
      sp.reserve_plan(4 + 3 + 6 + 6 + 8 + 18);  // BPix + FPix+ + TIB + TID+ + TOB + TEC+
      sp.fill_plan(0, 1, false, true);          // bk-fit only
      sp.append_plan(2, true);                  // pickup-only
      sp.append_plan(3, false);
      sp.append_plan(18, false);
      sp.append_plan(19, false);
      sp.append_plan(20, false);
      sp.fill_plan(4, 9);    // TIB,  6 layers (4 mono + 2 stereo)
      sp.fill_plan(21, 26);  // TID,  6 disks  (3 mono + 3 stereo)
      sp.fill_plan(10, 17);  // TOB,  8 layers (6 mono + 2 stereo)
      sp.fill_plan(27, 44);  // TEC, 18 disks  (9 mono + 9 stereo)
      sp.finalize_plan();
    }
    {
      SteeringParams &sp = ic.m_steering_params[TrackerInfo::Reg_Endcap_Pos];
      sp.reserve_plan(3 + 3 + 6 + 18);  // BPix + FPix+ + TID+ + TEC+; BPix4 is out of acceptance
      sp.fill_plan(0, 1, false, true);  // bk-fit only
      sp.append_plan(2, true);          // pickup-only
      sp.append_plan(18, false);
      sp.append_plan(19, false);
      sp.append_plan(20, false);
      sp.fill_plan(21, 26);  // TID,  6 disks  (3 mono + 3 stereo)
      sp.fill_plan(27, 44);  // TEC, 18 disks  (9 mono + 9 stereo)
      sp.finalize_plan();
    }
  }

  void partitionSeeds0(const TrackerInfo &trk_info,
                       const TrackVec &in_seeds,
                       const EventOfHits &eoh,
                       IterationSeedPartition &part) {
    // Seeds are placed into eta regions and sorted on region + eta.

    const int size = in_seeds.size();

    for (int i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const bool z_dir_pos = S.pz() > 0;

      HitOnTrack hot = S.getLastHitOnTrack();
      // MIMI ACHTUNG -- here we assume seed hits have already been remapped.
      // This was true at that time :)
      float eta = eoh[hot.layer].GetHit(hot.index).eta();
      // float  eta = S.momEta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      // Hardcoded for cms ... needs some lists of layers (hit/miss) for brl / ecp tests.
      // MM: Check lambda functions/std::function
      const LayerInfo &outer_brl = trk_info.outer_barrel_layer();

      const LayerInfo &tib1 = trk_info.m_layers[4];
      const LayerInfo &tob1 = trk_info.m_layers[10];

      const LayerInfo &tecp1 = trk_info.m_layers[27];
      const LayerInfo &tecn1 = trk_info.m_layers[54];

      const LayerInfo &tec_first = z_dir_pos ? tecp1 : tecn1;

      // If a track hits outer barrel ... it is in the barrel (for central, "outgoing" tracks).
      // This is also true for cyl-cow.
      // Better check is: hits outer TIB, misses inner TEC (but is +-z dependant).
      // XXXX Calculate z ... then check is inside or less that first EC z.
      // There are a lot of tracks that go through that crack.

      // XXXX trying a fix for low pT tracks that are in barrel after half circle
      float maxR = S.maxReachRadius();
      float z_at_maxr;

      bool can_reach_outer_brl = S.canReachRadius(outer_brl.m_rout);
      float z_at_outer_brl;
      bool misses_first_tec;
      if (can_reach_outer_brl) {
        z_at_outer_brl = S.zAtR(outer_brl.m_rout);
        if (z_dir_pos)
          misses_first_tec = z_at_outer_brl < tec_first.m_zmin;
        else
          misses_first_tec = z_at_outer_brl > tec_first.m_zmax;
      } else {
        z_at_maxr = S.zAtR(maxR);
        if (z_dir_pos)
          misses_first_tec = z_at_maxr < tec_first.m_zmin;
        else
          misses_first_tec = z_at_maxr > tec_first.m_zmax;
      }

      if (/*can_reach_outer_brl &&*/ misses_first_tec)
      // outer_brl.is_within_z_limits(S.zAtR(outer_brl.r_mean())))
      {
        reg = TrackerInfo::Reg_Barrel;
      } else {
        // This should be a list of layers
        // CMS, first tib, tob: 4, 10

        if ((S.canReachRadius(tib1.m_rin) && tib1.is_within_z_limits(S.zAtR(tib1.m_rin))) ||
            (S.canReachRadius(tob1.m_rin) && tob1.is_within_z_limits(S.zAtR(tob1.m_rin)))) {
          // transition region ... we are still hitting barrel layers

          reg = z_dir_pos ? TrackerInfo::Reg_Transition_Pos : TrackerInfo::Reg_Transition_Neg;
        } else {
          // endcap ... no barrel layers will be hit anymore.

          reg = z_dir_pos ? TrackerInfo::Reg_Endcap_Pos : TrackerInfo::Reg_Endcap_Neg;
        }
      }

      part.m_region[i] = reg;
      part.m_sort_score[i] = 5.0f * (reg - 2) + eta;
    }
  }
}  // namespace

namespace mkfit {
  void createPhase1TrackerGeometry(TrackerInfo &ti, IterationsInfo &ii, bool verbose) {
    // TODO: these writes to global variables need to be removed
    Config::nTotalLayers = 18 + 2 * 27;

    Config::useCMSGeom = true;

    Config::finding_requires_propagation_to_hit_pos = true;
    Config::finding_inter_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::finding_intra_layer_pflags = PropagationFlags(PF_none);
    Config::backward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::forward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    Config::seed_fit_pflags = PropagationFlags(PF_none);
    Config::pca_prop_pflags = PropagationFlags(PF_none);

    ti.set_eta_regions(0.9, 1.7, 2.45, false);
    ti.create_layers(18, 27, 27);

    ii.resize(9);
    ii[0].set_iteration_index_and_track_algorithm(0, (int)TrackBase::TrackAlgorithm::initialStep);
    ii[0].set_num_regions_layers(5, 72);

    createPhase1TrackerGeometryAutoGen(ti, ii);

    setupSteeringParamsIter0(ii[0]);
    ii[0].m_partition_seeds = partitionSeeds0;

    ii[1].Clone(ii[0]);
    ii[1].set_iteration_index_and_track_algorithm(1, (int)TrackBase::TrackAlgorithm::highPtTripletStep);

    ii[2].Clone(ii[0]);
    ii[2].set_iteration_index_and_track_algorithm(2, (int)TrackBase::TrackAlgorithm::lowPtQuadStep);

    ii[3].Clone(ii[0]);
    ii[3].set_iteration_index_and_track_algorithm(3, (int)TrackBase::TrackAlgorithm::lowPtTripletStep);

    ii[4].Clone(ii[0]);
    ii[4].set_iteration_index_and_track_algorithm(4, (int)TrackBase::TrackAlgorithm::detachedQuadStep);

    ii[5].Clone(ii[0]);
    ii[5].set_iteration_index_and_track_algorithm(5, (int)TrackBase::TrackAlgorithm::detachedTripletStep);

    ii[6].Clone(ii[0]);
    ii[6].set_iteration_index_and_track_algorithm(6, (int)TrackBase::TrackAlgorithm::mixedTripletStep);

    ii[7].Clone(ii[0]);
    ii[7].set_iteration_index_and_track_algorithm(7, (int)TrackBase::TrackAlgorithm::pixelLessStep);

    ii[8].Clone(ii[0]);
    ii[8].set_iteration_index_and_track_algorithm(8, (int)TrackBase::TrackAlgorithm::tobTecStep);

    //for the latter 2 iter investing in maxCand & stop condition (for time) + QF and Dupl. cleaning (for quality)

    // TODO: replace with MessageLogger
    if (verbose) {
      printf("==========================================================================================\n");
      printf("Phase1 tracker -- Create_TrackerInfo finished\n");
      printf("==========================================================================================\n");
      for (auto &i : ti.m_layers)
        i.print_layer();
      printf("==========================================================================================\n");
    }
  }
}  // namespace mkfit
