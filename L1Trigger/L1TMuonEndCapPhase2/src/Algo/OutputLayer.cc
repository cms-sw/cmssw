#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DataUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/OutputLayer.h"

using namespace emtf::phase2;
using namespace emtf::phase2::algo;

OutputLayer::OutputLayer(const EMTFContext& context) : context_(context) {}

void OutputLayer::apply(const int& endcap,
                        const int& sector,
                        const int& bx,
                        const std::map<int, int>& seg_to_hit,
                        const std::vector<track_t>& tracks,
                        const bool& displaced_en,
                        EMTFTrackCollection& out_tracks) const {
  const int endcap_pm = (endcap == 2) ? -1 : endcap;  // 1: +endcap, -1: -endcap

  for (auto& track : tracks) {  // Begin loop tracks
    // Fill Site/Hit Vectors
    int hit_count = 0;

    EMTFTrack::site_hits_t site_hits;
    EMTFTrack::site_segs_t site_segs;
    EMTFTrack::site_mask_t site_mask;
    EMTFTrack::site_mask_t site_rm_mask;

    for (unsigned int i = 0; i < v3::kNumTrackSites; i++) {
      // Get attached segments
      const auto& site_seg_id = track.site_segs[i];
      const auto& site_bit = track.site_mask[i];
      const auto& site_rm_bit = track.site_rm_mask[i];

      // Increase hit count
      if (site_bit == 1) {
        hit_count += 1;
      }

      // Convert segment to hit
      int hit_id = 0;

      if ((site_bit == 1) || (site_rm_bit == 1)) {
        hit_id = seg_to_hit.at(site_seg_id);
      }

      // Save Info
      site_hits.push_back(hit_id);
      site_segs.push_back(site_seg_id);
      site_mask.push_back(site_bit);
      site_rm_mask.push_back(site_rm_bit);
    }

    // Short-Circuit: Only keep tracks with hits
    if (!track.valid && hit_count == 0) {
      continue;
    }

    // Fill Feature Vector
    EMTFTrack::features_t model_features;

    for (unsigned int i = 0; i < v3::kNumTrackFeatures; i++) {
      model_features.push_back(track.features[i]);
    }

    // Find EMTF/GMT variables
    const int emtf_mode_v1 = findEMTFModeV1(track.site_mask);
    const int emtf_mode_v2 = findEMTFModeV2(track.site_mask);
    const int emtf_quality = findEMTFQuality(track, emtf_mode_v1, emtf_mode_v2);

    // Init Parameters
    auto& out_trk = out_tracks.emplace_back();

    out_trk.setEndcap(endcap_pm);
    out_trk.setSector(sector);
    out_trk.setBx(bx);
    out_trk.setUnconstrained(displaced_en ? true : false);
    out_trk.setValid(track.valid);

    out_trk.setModelPtAddress(track.pt_address);
    out_trk.setModelRelsAddress(track.rels_address);
    out_trk.setModelDxyAddress(track.dxy_address);
    out_trk.setModelPattern(track.pattern);
    out_trk.setModelQual(track.quality);
    out_trk.setModelPhi(track.phi);
    out_trk.setModelEta(track.theta);
    out_trk.setModelFeatures(model_features);

    out_trk.setEmtfQ(track.q);
    out_trk.setEmtfPt(track.pt);
    out_trk.setEmtfRels(track.rels);
    out_trk.setEmtfD0(std::abs(track.dxy));
    out_trk.setEmtfZ0(0);    // not yet implemented
    out_trk.setEmtfBeta(0);  // not yet implemented
    out_trk.setEmtfModeV1(emtf_mode_v1);
    out_trk.setEmtfModeV2(emtf_mode_v2);
    out_trk.setEmtfQuality(emtf_quality);

    out_trk.setSiteHits(site_hits);
    out_trk.setSiteSegs(site_segs);
    out_trk.setSiteMask(site_mask);
    out_trk.setSiteRMMask(site_rm_mask);
  }  // End loop tracks
}

int OutputLayer::findEMTFModeV1(const track_t::site_mask_t& x) const {
  int mode = 0;

  if (x[0] or x[9] or x[1] or x[5] or x[11]) {  // ME1/1, GE1/1, ME1/2, RE1/2, ME0
    mode |= (1 << 3);
  }

  if (x[2] or x[10] or x[6]) {  // ME2, GE2/1, RE2/2
    mode |= (1 << 2);
  }

  if (x[3] or x[7]) {  // ME3, RE3
    mode |= (1 << 1);
  }

  if (x[4] or x[8]) {  // ME4, RE4
    mode |= (1 << 0);
  }

  return mode;
}

// SingleMu (12)
// - at least one station-1 segment (ME1/1, GE1/1, ME1/2, RE1/2, ME0)
//   with one of the following requirements on stations 2,3,4
//   a. if there is ME1/2 or RE1/2,
//      i.  if there is ME1/2, require 1 more CSC station
//      ii. else, require 1 more CSC station + 1 more station
//   b. if there is ME1/1 or GE1/1,
//      i.  if there is ME1/1, require 1 more CSC station + 1 more station
//      ii. else, require 2 more CSC stations
//   c. if there is ME0,
//      i.  if there is ME1/1, require 1 more station in stations 3,4
//      ii. else, require 1 more CSC station + 1 more station
//
// DoubleMu (8)
// - at least one station-1 segment (ME1/1, GE1/1, ME1/2, RE1/2, ME0)
//   with one of the following requirements on stations 2,3,4
//   a. if there is ME1/1 or ME1/2, require 1 more station
//   b. if there is GE1/1 or RE1/2, require 1 more CSC station
//   c. if there is ME0,
//      i.  if there is ME1/1, require 1 more station
//      ii. else, require 1 more CSC station
//
// TripleMu (4)
// - at least two stations
//   a. if there is ME1/1 or ME1/2, require 1 more station
//   b. if there is GE1/1 or RE1/2, require 1 more CSC station
//   c. if there is ME0,
//      i.  if there is ME1/1, require 1 more station
//      ii. else, require 1 more CSC station
//   d. else, require 2 more CSC stations
//
// SingleHit (0)
// - at least one station
//
// Note that SingleMu, DoubleMu, TripleMu, SingleHit are mutually-exclusive categories.
int OutputLayer::findEMTFModeV2(const track_t::site_mask_t& x) const {
  int mode = 0;
  int cnt_ye11 = x[0] + x[9];                                          // ME1/1, GE1/1
  int cnt_ye12 = x[1] + x[5];                                          // ME1/2, RE1/2
  int cnt_ye22 = x[2] + x[10] + x[6];                                  // ME2, GE2/1, RE2/2
  int cnt_ye23 = x[3] + x[7];                                          // ME3, RE3
  int cnt_ye24 = x[4] + x[8];                                          // ME4, RE4
  int cnt_ye2a = (cnt_ye22 != 0) + (cnt_ye23 != 0) + (cnt_ye24 != 0);  //
  int cnt_ye2b = (cnt_ye23 != 0) + (cnt_ye24 != 0);                    //
  int cnt_me11 = x[0];                                                 // ME1/1 only
  int cnt_me12 = x[1];                                                 // ME1/2 only
  int cnt_me14 = x[11];                                                // ME0 only
  int cnt_me2a = (x[2] != 0) + (x[3] != 0) + (x[4] != 0);              //

  // SingleMu (12)
  {
    bool rule_a_i = (cnt_me12 != 0) and (cnt_me2a >= 1);
    bool rule_a_ii = (cnt_ye12 != 0) and (cnt_me2a >= 1) and (cnt_ye2a >= 2);
    bool rule_b_i = (cnt_me11 != 0) and (cnt_me2a >= 1) and (cnt_ye2a >= 2);
    bool rule_b_ii = (cnt_ye11 != 0) and (cnt_me2a >= 2);
    bool rule_c_i = (cnt_me14 != 0) and (cnt_me11 != 0) and (cnt_ye2b >= 1);
    bool rule_c_ii = (cnt_me14 != 0) and (cnt_me2a >= 1) and (cnt_ye2a >= 2);

    if (rule_a_i or rule_a_ii or rule_b_i or rule_b_ii or rule_c_i or rule_c_ii) {
      mode |= (1 << 3);
      mode |= (1 << 2);
    }
  }

  // DoubleMu (8)
  if (mode < (1 << 3)) {
    bool rule_a_i = (cnt_me12 != 0) and (cnt_ye2a >= 1);
    bool rule_a_ii = (cnt_me11 != 0) and (cnt_ye2a >= 1);
    bool rule_b_i = (cnt_ye12 != 0) and (cnt_me2a >= 1);
    bool rule_b_ii = (cnt_ye11 != 0) and (cnt_me2a >= 1);
    bool rule_c_i = (cnt_me14 != 0) and (cnt_me11 != 0) and (cnt_ye2a >= 1);
    bool rule_c_ii = (cnt_me14 != 0) and (cnt_me2a >= 1);

    if (rule_a_i or rule_a_ii or rule_b_i or rule_b_ii or rule_c_i or rule_c_ii) {
      mode |= (1 << 3);
    }
  }

  // TripleMu (4)
  if (mode < (1 << 2)) {
    bool rule_a_i = (cnt_me12 != 0) and (cnt_ye2a >= 1);
    bool rule_a_ii = (cnt_me11 != 0) and (cnt_ye2a >= 1);
    bool rule_b_i = (cnt_ye12 != 0) and (cnt_me2a >= 1);
    bool rule_b_ii = (cnt_ye11 != 0) and (cnt_me2a >= 1);
    bool rule_c_i = (cnt_me14 != 0) and (cnt_me11 != 0) and (cnt_ye2a >= 1);
    bool rule_c_ii = (cnt_me14 != 0) and (cnt_me2a >= 1);
    bool rule_d = (cnt_me2a >= 2);

    if (rule_a_i or rule_a_ii or rule_b_i or rule_b_ii or rule_c_i or rule_c_ii or rule_d) {
      mode |= (1 << 2);
    }
  }

  return mode;
}

int OutputLayer::findEMTFQuality(const track_t& track, const int& mode_v1, const int& mode_v2) const {
  // Short-Circuit: Invalid track
  if (track.valid == 0) {
    return 0;
  }

  // Short-Circuit: Single Station
  bool is_single_station = (mode_v1 == 0);
  is_single_station |= (mode_v1 == 1);
  is_single_station |= (mode_v1 == 2);
  is_single_station |= (mode_v1 == 4);
  is_single_station |= (mode_v1 == 8);

  if (is_single_station) {
    return 0;
  }

  // Calculate Quality Based on ModeV2
  if (mode_v2 == 0) {
    // Single Hit
    if ((0 <= track.quality) && (track.quality <= 3)) {
      return 1;
    } else if ((4 <= track.quality) && (track.quality <= 7)) {
      return 2;
    } else if (7 < track.quality) {
      return 3;
    }

    return 0;
  } else if (mode_v2 == 4) {
    // Triple Muon Quality
    if ((8 <= track.quality) && (track.quality <= 11)) {
      return 5;
    } else if ((12 <= track.quality) && (track.quality <= 15)) {
      return 6;
    } else if (15 < track.quality) {
      return 7;
    }

    return 4;
  } else if (mode_v2 == 8) {
    // Double Muon Quality
    bool valid_mode = (mode_v1 == 9);
    valid_mode |= (mode_v1 == 10);
    valid_mode |= (mode_v1 == 12);

    if (valid_mode) {
      if ((16 <= track.quality) && (track.quality <= 23)) {
        return 9;
      } else if ((24 <= track.quality) && (track.quality <= 31)) {
        return 10;
      } else if (31 < track.quality) {
        return 11;
      }
    }

    return 8;
  } else if (mode_v2 == 12) {
    // Single Muon Quality
    bool valid_mode = (mode_v1 == 11);
    valid_mode |= (mode_v1 == 13);
    valid_mode |= (mode_v1 == 14);
    valid_mode |= (mode_v1 == 15);

    if (valid_mode) {
      if ((32 <= track.quality) && (track.quality <= 39)) {
        return 13;
      } else if ((40 <= track.quality) && (track.quality <= 51)) {
        return 14;
      } else if (51 < track.quality) {
        return 15;
      }
    }

    return 12;
  }

  // Invalid track
  return 0;
}
