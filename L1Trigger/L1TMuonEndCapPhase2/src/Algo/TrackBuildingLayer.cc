#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DataUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/TrackBuildingLayer.h"

using namespace emtf::phase2;
using namespace emtf::phase2::algo;

// Static
seg_theta_t TrackBuildingLayer::calc_theta_median(std::vector<seg_theta_t> thetas) {
  auto i_last = thetas.size() - 1;

  // Sort Thetas
  // This will sort ascending order (lower-value means lower-index)
  data::mergesort(&thetas[0], thetas.size(), [](seg_theta_t lower_index_value, seg_theta_t larger_index_value) -> int {
    return lower_index_value > larger_index_value ? 1 : 0;
  });

  // Check if any model_thm_site is null
  // Since the theta array has been sorted, it's enough
  // to check the last index, because the invalid value will be the max
  seg_theta_t invalid_theta = -1;  // This maps to 255 since it underflows

  bool any_invalid = thetas[i_last] == invalid_theta;

  // Calculate median
  if (any_invalid) {
    // Use the min value as the median if there are any invalid thetas
    return thetas[0];
  } else {
    // Calculate the median if all thetas are valid
    return data::median_of_sorted(&thetas[0], thetas.size());
  }
}

// Members
TrackBuildingLayer::TrackBuildingLayer(const EMTFContext& context) : context_(context) {}

void TrackBuildingLayer::apply(const segment_collection_t& segments,
                               const std::vector<road_t>& roads,
                               const bool& displaced_en,
                               std::vector<track_t>& tracks) const {
  // Apply
  for (unsigned int i_road = 0; i_road < roads.size(); ++i_road) {
    // Get road and track
    const auto& road = roads[i_road];
    auto& track = tracks.emplace_back();

    // Initialize track
    track.phi = 0;
    track.theta = 0;
    track.valid = 0;

    for (int site_id = 0; site_id < v3::kNumTrackSites; ++site_id) {
      track.site_segs[site_id] = 0;
      track.site_mask[site_id] = 0;
      track.site_rm_mask[site_id] = 0;
    }

    for (int i_feature = 0; i_feature < v3::kNumTrackFeatures; ++i_feature) {
      track.features[i_feature] = 0;
    }

    // Short-Circuit: If the road has quality-0 skip it
    if (road.quality == 0) {
      continue;
    }

    // Debug Info
    if (this->context_.config_.verbosity_ > 1) {
      if (i_road == 0) {
        edm::LogInfo("L1TEMTFpp") << std::endl;
        edm::LogInfo("L1TEMTFpp") << "==========================================================================="
                                  << std::endl;
        edm::LogInfo("L1TEMTFpp") << "BEGIN TRACK BUILDING" << std::endl;
        edm::LogInfo("L1TEMTFpp") << "---------------------------------------------------------------------------"
                                  << std::endl;
      }

      edm::LogInfo("L1TEMTFpp") << "***************************************************************************"
                                << std::endl;
      edm::LogInfo("L1TEMTFpp") << "Begin building track " << i_road << std::endl;
    }

    // Attach segments
    attach_segments(segments, road, displaced_en, track);

    // Debug Info
    if (this->context_.config_.verbosity_ > 1) {
      edm::LogInfo("L1TEMTFpp") << "End building track " << i_road << std::endl;

      if (i_road == (roads.size() - 1)) {
        edm::LogInfo("L1TEMTFpp") << "---------------------------------------------------------------------------"
                                  << std::endl;
        edm::LogInfo("L1TEMTFpp") << "END TRACK BUILDING" << std::endl;
        edm::LogInfo("L1TEMTFpp") << "==========================================================================="
                                  << std::endl;
      }
    }
  }
}

void TrackBuildingLayer::attach_segments(const segment_collection_t& segments,
                                         const road_t& road,
                                         const bool& displaced_en,
                                         track_t& track) const {
  // ===========================================================================
  // Constants
  // ---------------------------------------------------------------------------
  seg_theta_t invalid_theta = -1;  // This will map to 255 since it underflows

  // ===========================================================================
  // Unpack road
  // ---------------------------------------------------------------------------
  // trk_col: Recall that the hitmap is 288 cols wide, and the full chamber hitmap is 315 cols;
  // the chamber hitmap doesn't fit in the hitmap, so we skipped the first 27 cols.
  // In order to calculate the full hitmap col, we need to add back the 27 cols that we skipped.
  // sector_col: The sector's column is the center col of the phi map adding back the 27 skipped cols.
  const auto trk_zone = road.zone;
  const auto trk_pattern = road.pattern;
  const auto trk_quality = road.quality;

  int bit_sel_zone = (1u << trk_zone);

  const trk_col_t trk_col = road.col + v3::kHitmapCropColStart;
  const trk_col_t sector_col = static_cast<trk_col_t>(v3::kHitmapNCols / 2) + v3::kHitmapCropColStart;

  // ===========================================================================
  // Initialize vars
  // ---------------------------------------------------------------------------
  std::array<seg_phi_t, v3::kNumTrackSites> trk_seg_phi_diff;
  std::array<seg_theta_t, v3::kNumTrackSites> trk_seg_theta;

  for (int site_id = 0; site_id < v3::kNumTrackSites; ++site_id) {
    trk_seg_phi_diff[site_id] = 0;
    trk_seg_theta[site_id] = 0;
  }

  // ===========================================================================
  // Unpack model
  // ---------------------------------------------------------------------------
  const auto& model = context_.model_;
  const auto& model_hm = model.zones_[trk_zone].hitmap;
  const auto& model_ftc = model.features_;

  auto* model_pat = &(model.zones_[trk_zone].prompt_patterns[trk_pattern]);

  if (displaced_en) {
    model_pat = &(model.zones_[trk_zone].disp_patterns[trk_pattern]);
  }

  // ===========================================================================
  // Convert column center to emtf_phi units
  // ---------------------------------------------------------------------------
  // Each column is emtf_phi=1<<n wide, therefore half of this would be 1<<(n-1)
  // since shifting to right m, is the same as dividing by 2^m.
  seg_phi_t trk_abs_phi =
      (static_cast<seg_phi_t>(trk_col) << v3::kHitmapColFactorLog2) + (1 << (v3::kHitmapColFactorLog2 - 1));
  seg_phi_t sector_abs_phi =
      (static_cast<seg_phi_t>(sector_col) << v3::kHitmapColFactorLog2) + (1 << (v3::kHitmapColFactorLog2 - 1));

  // Calculate track phi
  // Note this is the track phi with respect to the sector center
  trk_feature_t trk_rel_phi = static_cast<trk_feature_t>(trk_abs_phi) - static_cast<trk_feature_t>(sector_abs_phi);

  // ===========================================================================
  // Get pattern info for each row
  // ---------------------------------------------------------------------------
  std::array<trk_col_t, v3::kHitmapNRows> trk_pat_begin;
  std::array<trk_col_t, v3::kHitmapNRows> trk_pat_center;
  std::array<trk_col_t, v3::kHitmapNRows> trk_pat_end;
  std::array<seg_phi_t, v3::kHitmapNRows> trk_pat_phi;

  for (int i_row = 0; i_row < v3::kHitmapNRows; ++i_row) {
    // Get the model pattern
    const auto& model_pat_row = (*model_pat)[i_row];

    // Offset the pattern's begin, center, and end by the track column
    trk_pat_begin[i_row] = trk_col + model_pat_row.begin;
    trk_pat_center[i_row] = trk_col + model_pat_row.center;
    trk_pat_end[i_row] = trk_col + model_pat_row.end;
    trk_pat_phi[i_row] = 0;

    // Short-Circuit: If the pattern's center is less than the padding used
    // when matching the pattern to the hitmap then the pattern center is 0.
    // This is because at that point, the center is out-of-bounds.
    if (trk_pat_center[i_row] <= v3::kPatternMatchingPadding)
      continue;

    // When the center is beyond the padding, then the pattern
    // is in-bound, therefore we subtract the padding offset.
    // To get the center in terms of the non-padded row BW we need to remove padding
    // since col-padding + 1 should map to 1 in the non-padded hitmap
    const auto& temp_trk_pat_center = trk_pat_center[i_row] - v3::kPatternMatchingPadding;

    // Convert the pattern center to emtf_phi units
    trk_pat_phi[i_row] = (static_cast<seg_phi_t>(temp_trk_pat_center) << v3::kHitmapColFactorLog2) +
                         (1 << (v3::kHitmapColFactorLog2 - 1));
  }

  // ===========================================================================
  // Select segments using phi only
  // ---------------------------------------------------------------------------
  int n_rows = model_hm.size();

  // clang-format off
    std::vector<std::vector<unsigned int>> site_chambers = {
        {  0,   1,   2,   9,  10,  11,  45}, // ME1/1
        {  3,   4,   5,  12,  13,  14,  46}, // ME1/2
        { 18,  19,  20,  48,  21,  22,  23,  24,  25,  26,  49}, // ME2/1 + ME2/2
        { 27,  28,  29,  50,  30,  31,  32,  33,  34,  35,  51}, // ME3/1 + ME3/2
        { 36,  37,  38,  52,  39,  40,  41,  42,  43,  44,  53}, // ME4/1 + ME4/2
        { 57,  58,  59,  66,  67,  68, 100}, // RE1/2
        { 75,  76,  77,  78,  79,  80, 103}, // RE2/2
        { 81,  82,  83, 104,  84,  85,  86,  87,  88,  89, 105}, // RE3/1 + RE3/2
        { 90,  91,  92, 106,  93,  94,  95,  96,  97,  98, 107}, // RE4/1 + RE4/2
        { 54,  55,  56,  63,  64,  65,  99}, // GE1/1 
        { 72,  73,  74, 102}, // GE2/1
        {108, 109, 110, 111, 112, 113, 114} // ME0
    };

    std::vector<unsigned int> site_chamber_orders = {
        0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 1, 0
    };

    std::vector<std::vector<int>> chamber_orders = {
        {-1, -1,  6, -1,  0,  1, -1,  2,  3, -1,  4,  5},
        { 3, -1, -1,  0, -1, -1,  1, -1, -1,  2, -1, -1},
        { 3, -1, 10,  0,  4,  5,  1,  6,  7,  2,  8,  9}
    };
  // clang-format on

  for (int i_row = 0; i_row < n_rows; ++i_row) {  // Begin loop rows

    const auto& model_hm_row = model_hm[i_row];

    const auto& trk_pat_row_begin = trk_pat_begin[i_row];
    const auto& trk_pat_row_end = trk_pat_end[i_row];
    const auto& trk_pat_row_phi = trk_pat_phi[i_row];

    if (this->context_.config_.verbosity_ > 2) {
      edm::LogInfo("L1TEMTFpp") << "Pattern Row:"
                                << " row " << i_row << " begin " << trk_pat_row_begin << " end " << trk_pat_row_end
                                << " phi " << trk_pat_row_phi << std::endl;
    }

    for (const auto& model_hm_site : model_hm_row) {  // Begin loop sites in row

      const int site_id = static_cast<int>(model_hm_site.id);

      auto& site_seg_id = track.site_segs[site_id];
      auto& site_bit = track.site_mask[site_id];
      auto& site_min_phi_diff = trk_seg_phi_diff[site_id];

      const auto& s_chambers = site_chambers[site_id];
      const auto& s_chamber_order_id = site_chamber_orders[site_id];
      const auto& s_chamber_order = chamber_orders[s_chamber_order_id];

      for (const auto& chamber_idx : s_chamber_order) {  // Begin loop chambers in site

        if (chamber_idx == -1)
          continue;

        int chamber_id = s_chambers[chamber_idx];

        for (int i_ch_seg = 0; i_ch_seg < v3::kChamberSegments; ++i_ch_seg) {  // Begin loop segments

          const int seg_id = chamber_id * v3::kChamberSegments + i_ch_seg;
          const auto& seg = segments[seg_id];

          // Short-Circuit: If the segment is invalid move on
          if (!seg.valid) {
            continue;
          }

          // Short-Circuit: If the segment is not in the zone move on
          if ((seg.zones & bit_sel_zone) != bit_sel_zone) {
            continue;
          }

          // Short-Circuit: If the segment is outside of the pattern move on
          const trk_col_t seg_col = (seg.phi >> 4) + v3::kPatternMatchingPadding;

          if (!(trk_pat_row_begin <= seg_col && seg_col <= trk_pat_row_end)) {
            continue;
          }

          // Calculate abs diff between the pattern's row phi and the segment's phi
          seg_phi_t diff;

          if (trk_pat_row_phi > seg.phi) {
            diff = trk_pat_row_phi - seg.phi;
          } else {
            diff = seg.phi - trk_pat_row_phi;
          }

          if (this->context_.config_.verbosity_ > 2) {
            edm::LogInfo("L1TEMTFpp") << "Site candidate:"
                                      << " site_id " << site_id << " seg_id " << seg_id << " seg_phi " << seg.phi
                                      << " seg_theta1 " << seg.theta1 << " seg_theta2 " << seg.theta2 << " seg_bend "
                                      << seg.bend << std::endl;
          }

          // Short-Circuit: If the difference is larger than the min diff move on
          if (site_bit == 1 && site_min_phi_diff <= diff)
            continue;

          // Select better segment
          site_seg_id = seg_id;
          site_bit = 1;
          site_min_phi_diff = diff;
        }  // End loop segments

      }  // End loop chambers in site

      // Debug Info
      if (this->context_.config_.verbosity_ > 2 && site_bit == 1) {
        edm::LogInfo("L1TEMTFpp") << "Segment attached:"
                                  << " site_id " << site_id << " seg_id " << site_seg_id << " seg_phi "
                                  << segments[site_seg_id].phi << " seg_theta1 " << segments[site_seg_id].theta1
                                  << " seg_theta2 " << segments[site_seg_id].theta2 << " seg_bend "
                                  << segments[site_seg_id].bend << std::endl;
      }
    }  // End loop sites in row

  }  // End loop rows

  // ===========================================================================
  // Calculate theta medians
  // ---------------------------------------------------------------------------
  const auto& model_thmc = model.theta_medians_;

  std::vector<seg_theta_t> theta_medians;

  for (const auto& model_thm : model_thmc) {  // Begin loop model theta medians

    std::vector<seg_theta_t> group_medians;

    for (const auto& model_thm_group : model_thm) {  // Begin loop theta median groups

      std::vector<seg_theta_t> group;

      for (const auto& model_thm_site : model_thm_group) {  // Begin loop group sites
        int site_id = static_cast<int>(model_thm_site.id);

        const auto& site_bit = track.site_mask[site_id];

        // Initialize as invalid theta
        auto& theta = group.emplace_back(invalid_theta);

        // Short-Circuit: If no segment was selected, move on.
        if (site_bit == 0)
          continue;

        // Get selected segment's theta value
        const auto& site_seg_id = track.site_segs[site_id];
        const auto& site_seg = segments[site_seg_id];

        if (model_thm_site.theta_id == theta_id_t::kTheta1) {
          theta = site_seg.theta1;
        } else if (model_thm_site.theta_id == theta_id_t::kTheta2) {
          theta = site_seg.theta2;
        }

        // If the segment theta is 0 this is invalid theta value
        if (theta == 0) {
          theta = invalid_theta;
        }
      }  // End loop group sites

      // Calculate theta median
      if (this->context_.config_.verbosity_ > 2) {
        for (const auto& theta : group) {
          edm::LogInfo("L1TEMTFpp") << "theta " << theta << std::endl;
        }
      }

      auto group_median = calc_theta_median(group);
      group_medians.push_back(group_median);

      if (this->context_.config_.verbosity_ > 2) {
        edm::LogInfo("L1TEMTFpp") << "group_median " << group_median << std::endl;
      }
    }  // End loop theta median groups

    // Calculate model_thm_group median
    auto theta_median = calc_theta_median(group_medians);
    theta_medians.push_back(theta_median);

    if (this->context_.config_.verbosity_ > 2) {
      edm::LogInfo("L1TEMTFpp") << "theta_median " << theta_median << std::endl;
    }
  }  // End loop theta medians

  // ===========================================================================
  // Select track theta
  // ---------------------------------------------------------------------------
  seg_theta_t trk_abs_theta;

  if (trk_zone != 2) {
    trk_abs_theta = theta_medians[0];
  } else {
    trk_abs_theta = theta_medians[1];
  }

  // If median is invalid, try station 1 median
  if (trk_abs_theta == invalid_theta) {
    trk_abs_theta = theta_medians[2];
  }

  // If all medians are invalid use 0 (0 is an invalid theta)
  if (trk_abs_theta == invalid_theta) {
    trk_abs_theta = 0;
  }

  // ===========================================================================
  // Compare segment theta to track theta
  // ---------------------------------------------------------------------------

  // if theta_window < diff, it is invalid

  // clang-format off
    std::vector<std::vector<seg_theta_t>> site_theta_window = {
        {5, 0, 2, 2, 2, 34, 0, 3, 3, 5, 6, 5},
        {5, 9, 5, 4, 5, 14, 7, 7, 7, 7, 7, 4},
        {11, 6, 5, 6, 6, 10, 8, 8, 9, 8, 0, 0}
    };
  // clang-format on

  if (displaced_en) {
    // clang-format off
        site_theta_window = {
            {14, 40, 4, 3, 3, 45, 0, 4, 4, 15, 8, 13},
            {16, 18, 7, 5, 5, 22, 7, 7, 8, 17, 9, 14},
            {26, 15, 8, 9, 9, 17, 11, 9, 10, 26, 21, 0}
        };
    // clang-format on
  }

  for (int site_id = 0; site_id < v3::kNumTrackSites; ++site_id) {
    auto& site_bit = track.site_mask[site_id];
    auto& site_rm_bit = track.site_rm_mask[site_id];

    // Get Theta Window
    const auto& theta_window = site_theta_window[trk_zone][site_id];

    // Short-Circuit: If no segment was selected, move on.
    if (site_bit == 0)
      continue;

    const auto& site_seg_id = track.site_segs[site_id];
    const auto& site_seg = segments[site_seg_id];

    // Init differences with out-of-bounds values
    seg_theta_t diff_1 = theta_window + 1;
    seg_theta_t diff_2 = theta_window + 1;

    // Calculate abs theta 1 diff
    if (site_seg.theta1 != 0) {
      if (site_seg.theta1 < trk_abs_theta) {
        diff_1 = trk_abs_theta - site_seg.theta1;
      } else {
        diff_1 = site_seg.theta1 - trk_abs_theta;
      }
    }

    // Calculate abs theta 2 diff
    if (site_seg.theta2 != 0) {
      if (site_seg.theta2 < trk_abs_theta) {
        diff_2 = trk_abs_theta - site_seg.theta2;
      } else {
        diff_2 = site_seg.theta2 - trk_abs_theta;
      }
    }

    // Select the theta with the smallest difference
    if (diff_1 <= diff_2 && diff_1 < theta_window) {
      // Select theta 1 as the correct theta value
      trk_seg_theta[site_id] = site_seg.theta1;
    } else if (diff_2 < theta_window) {
      // Select theta 2 as the correct theta value
      trk_seg_theta[site_id] = site_seg.theta2;
    } else {
      // Invalidate site if both differences are outside of the theta window
      site_bit = 0;
      site_rm_bit = 1;

      // Debug Info
      if (this->context_.config_.verbosity_ > 4) {
        edm::LogInfo("L1TEMTFpp") << "Segment outside of theta window; detatched:"
                                  << " site_id " << site_id << " seg_id " << site_seg_id << " seg_phi " << site_seg.phi
                                  << " seg_theta1 " << site_seg.theta1 << " seg_theta2 " << site_seg.theta2
                                  << std::endl;
      }
    }
  }

  // ===========================================================================
  // Assign Data
  // ---------------------------------------------------------------------------
  track.zone = trk_zone;
  track.col = trk_col;
  track.pattern = trk_pattern;
  track.quality = trk_quality;
  track.phi = trk_abs_phi;
  track.theta = trk_abs_theta;
  track.valid = 1;

  // ===========================================================================
  // Fill features
  // ---------------------------------------------------------------------------
  int i_feature = 0;

  for (auto& model_ft : model_ftc) {
    for (auto& model_ft_site : model_ft.sites) {
      int site_id = static_cast<int>(model_ft_site);

      const auto& site_seg_id = track.site_segs[site_id];
      const auto& site_bit = track.site_mask[site_id];
      const auto& site_seg = segments[site_seg_id];

      auto& trk_feature = track.features[i_feature++];

      // Short-Circuit: No segment attached
      if (site_bit == 0) {
        continue;
      }

      // Fill features
      if (model_ft.id == feature_id_t::kPhi) {
        // Note: This is the segment's phi with respect to the track's abs phi
        trk_feature = static_cast<trk_feature_t>(site_seg.phi) - static_cast<trk_feature_t>(trk_abs_phi);
      } else if (model_ft.id == feature_id_t::kTheta) {
        // Note: This is the segment's theta with respect to the track's abs theta
        trk_feature = static_cast<trk_feature_t>(trk_seg_theta[site_id]) - static_cast<trk_feature_t>(trk_abs_theta);
      } else if (model_ft.id == feature_id_t::kBend) {
        trk_feature = site_seg.bend;
      } else if (model_ft.id == feature_id_t::kQuality) {
        trk_feature = site_seg.qual1;
      }
    }
  }

  // Additional features
  track.features[i_feature++] = trk_quality > 0 ? trk_rel_phi : decltype(trk_rel_phi)(0);
  track.features[i_feature++] = trk_quality > 0 ? trk_abs_theta : decltype(trk_abs_theta)(0);
  track.features[i_feature++] = trk_quality;
  track.features[i_feature++] = 0;  // unused

  // Debug Info
  if (this->context_.config_.verbosity_ > 1) {
    edm::LogInfo("L1TEMTFpp") << "Track"
                              << " zone " << track.zone << " col " << track.col << " pat " << track.pattern << " qual "
                              << track.quality << " sector_abs_phi " << sector_abs_phi << " abs_phi " << track.phi
                              << " rel_phi " << trk_rel_phi << " abs_theta " << track.theta << " features "
                              << std::endl;

    for (int i = 0; i < v3::kNumTrackFeatures; ++i) {
      if (i > 0) {
        edm::LogInfo("L1TEMTFpp") << " ";
      }

      edm::LogInfo("L1TEMTFpp") << track.features[i];
    }

    edm::LogInfo("L1TEMTFpp") << std::endl;
  }
}
