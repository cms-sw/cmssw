#include "L1Trigger/L1TMuonEndCap/interface/AngleCalculation.h"

#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"
#include "helper.h"  // to_hex, to_binary

namespace {
  static const int bw_fph = 13;  // bit width of ph, full precision
  static const int bw_th = 7;    // bit width of th
  static const int invalid_dtheta = (1<<bw_th) - 1;  // = 127
  static const int invalid_dphi = (1<<bw_fph) - 1;   // = 8191
}


void AngleCalculation::configure(
    int verbose, int endcap, int sector, int bx,
    int bxWindow,
    int thetaWindow, int thetaWindowRPC,
    bool bugME11Dupes
) {
  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  bxWindow_        = bxWindow;
  thetaWindow_     = thetaWindow;
  thetaWindowRPC_  = thetaWindowRPC;
  bugME11Dupes_    = bugME11Dupes;
}

void AngleCalculation::process(
    zone_array<EMTFTrackCollection>& zone_tracks
) const {

  for (int izone = 0; izone < NUM_ZONES; ++izone) {
    EMTFTrackCollection& tracks = zone_tracks.at(izone);  // pass by reference

    EMTFTrackCollection::iterator tracks_it  = tracks.begin();
    EMTFTrackCollection::iterator tracks_end = tracks.end();

    // Calculate deltas
    for (; tracks_it != tracks_end; ++tracks_it) {
      calculate_angles(*tracks_it);
    }

    // Erase tracks with rank = 0
    // Erase hits that are not selected as the best phi and theta in each station
    erase_tracks(tracks);

    tracks_it  = tracks.begin();
    tracks_end = tracks.end();

    // Calculate bx
    // (in the firmware, this happens during best track selection.)
    for (; tracks_it != tracks_end; ++tracks_it) {
      calculate_bx(*tracks_it);
    }
  }  // end loop over zones

  if (verbose_ > 0) {  // debug
    for (const auto& tracks : zone_tracks) {
      for (const auto& track : tracks) {
        std::cout << "deltas: z: " << track.Zone()-1 << " pat: " << track.Winner() << " rank: " << to_hex(track.Rank())
            << " delta_ph: " << array_as_string(track.PtLUT().delta_ph)
            << " delta_th: " << array_as_string(track.PtLUT().delta_th)
            << " sign_ph: " << array_as_string(track.PtLUT().sign_ph)
            << " sign_th: " << array_as_string(track.PtLUT().sign_th)
            << " phi: " << track.Phi_fp()
            << " theta: " << track.Theta_fp()
            << " cpat: " << array_as_string(track.PtLUT().cpattern)
            << " v: " << array_as_string(track.PtLUT().bt_vi)
            << " h: " << array_as_string(track.PtLUT().bt_hi)
            << " c: " << array_as_string(track.PtLUT().bt_ci)
            << " s: " << array_as_string(track.PtLUT().bt_si)
            << std::endl;
      }
    }
  }

}

void AngleCalculation::calculate_angles(EMTFTrack& track) const {
  // Group track hits by station
  std::array<EMTFHitCollection, NUM_STATIONS> st_conv_hits;

  for (int istation = 0; istation < NUM_STATIONS; ++istation) {
    for (const auto& conv_hit : track.Hits()) {
      if ((conv_hit.Station() - 1) == istation) {
        st_conv_hits.at(istation).push_back(conv_hit);
      }
    }

    if (bugME11Dupes_)
      assert(st_conv_hits.at(istation).size() <= 4);  // ambiguity in theta is max 4
    else
      assert(st_conv_hits.at(istation).size() <= 2);  // ambiguity in theta is max 2
  }
  assert(st_conv_hits.size() == NUM_STATIONS);


  // Best theta deltas and phi deltas
  // from 0 to 5: dtheta12, dtheta13, dtheta14, dtheta23, dtheta24, dtheta34
  std::array<int,  NUM_STATION_PAIRS> best_dtheta_arr;  // Best of up to 4 dTheta values per pair of stations (with duplicate thetas)
  std::array<int,  NUM_STATION_PAIRS> best_dtheta_sign_arr;
  std::array<int,  NUM_STATION_PAIRS> best_dphi_arr;   // Not really "best" - there is only one dPhi value per pair of stations
  std::array<int,  NUM_STATION_PAIRS> best_dphi_sign_arr;

  // Best angles
  // from 0 to 5: ME2,      ME3,      ME4,      ME2,      ME2,      ME3
  //              dtheta12, dtheta13, dtheta14, dtheta23, dtheta24, dtheta34
  std::array<int,  NUM_STATION_PAIRS> best_theta_arr;
  std::array<int,  NUM_STATION_PAIRS> best_phi_arr;

  // Keep track of which pair is valid
  std::array<bool, NUM_STATION_PAIRS> best_dtheta_valid_arr;
  std::array<bool, NUM_STATION_PAIRS> best_has_rpc_arr; // Not used - should remove (AWB 21.05.17)

  // Initialize
  best_dtheta_arr      .fill(invalid_dtheta);
  best_dtheta_sign_arr .fill(1);
  best_dphi_arr        .fill(invalid_dphi);
  best_dphi_sign_arr   .fill(1);
  best_phi_arr         .fill(0);
  best_theta_arr       .fill(0);
  best_dtheta_valid_arr.fill(false);
  best_has_rpc_arr     .fill(false);

  auto abs_diff = [](int a, int b) { return std::abs(a-b); };

  // Calculate angles
  int ipair = 0;

  for (int ist1 = 0; ist1+1 < NUM_STATIONS; ++ist1) {  // station A
    for (int ist2 = ist1+1; ist2 < NUM_STATIONS; ++ist2) {  // station B
      const EMTFHitCollection& conv_hitsA = st_conv_hits.at(ist1);
      const EMTFHitCollection& conv_hitsB = st_conv_hits.at(ist2);

      // More than 1 hit per station when hit has ambigous theta
      for (const auto& conv_hitA : conv_hitsA) {
        for (const auto& conv_hitB : conv_hitsB) {
          // Has RPC?
          //bool has_rpc = (conv_hitA.Subsystem() == TriggerPrimitive::kRPC || conv_hitB.Subsystem() == TriggerPrimitive::kRPC);

          // Calculate theta deltas
          int thA = conv_hitA.Theta_fp();
          int thB = conv_hitB.Theta_fp();
          int dth = abs_diff(thA, thB);
          int dth_sign = (thA <= thB);  // sign
          assert(thA != 0 && thB != 0);
          assert(dth < invalid_dtheta);

          if (best_dtheta_arr.at(ipair) >= dth) {
            best_dtheta_arr.at(ipair) = dth;
            best_dtheta_sign_arr.at(ipair) = dth_sign;
            best_dtheta_valid_arr.at(ipair) = true;
            // best_has_rpc_arr.at(ipair) = has_rpc;  // FW doesn't check whether a segment is CSC or RPC, should remove - AWB 21.05.17

            // first 3 pairs, use station B
            // last 3 pairs, use station A
            best_theta_arr.at(ipair) = (ipair < 3) ? thB : thA;
          }

          // Calculate phi deltas
          int phA = conv_hitA.Phi_fp();
          int phB = conv_hitB.Phi_fp();
          int dph = abs_diff(phA, phB);
          int dph_sign = (phA <= phB);

          if (best_dphi_arr.at(ipair) >= dph) {
            best_dphi_arr.at(ipair) = dph;
            best_dphi_sign_arr.at(ipair) = dph_sign;

            // first 3 pairs, use station B
            // last 3 pairs, use station A
            best_phi_arr.at(ipair) = (ipair < 3) ? phB : phA;
          }
        }  // end loop over conv_hits in station B
      }  // end loop over conv_hits in station A

      ++ipair;
    }  // end loop over station B
  }  // end loop over station A
  assert(ipair == NUM_STATION_PAIRS);


  // Apply cuts on dtheta

  // There is a possible bug in FW. After a dtheta pair fails the theta window
  // cut, the valid flag of the pair is not updated. Later on, theta from
  // this pair is used to assign the precise theta of the track.
  std::array<bool, NUM_STATION_PAIRS> best_dtheta_valid_arr_1;

  for (int ipair = 0; ipair < NUM_STATION_PAIRS; ++ipair) {
    if (best_has_rpc_arr.at(ipair))
      best_dtheta_valid_arr_1.at(ipair) = best_dtheta_valid_arr.at(ipair) && (best_dtheta_arr.at(ipair) <= thetaWindowRPC_);
    else
      best_dtheta_valid_arr_1.at(ipair) = best_dtheta_valid_arr.at(ipair) && (best_dtheta_arr.at(ipair) <= thetaWindow_);
  }

  // Find valid segments
  // vmask contains valid station mask = {ME4,ME3,ME2,ME1}. "0b" prefix for binary.
  int vmask1 = 0, vmask2 = 0, vmask3 = 0;

  if (best_dtheta_valid_arr_1.at(0)) {
    vmask1 |= 0b0011;  // 12
  }
  if (best_dtheta_valid_arr_1.at(1)) {
    vmask1 |= 0b0101;  // 13
  }
  if (best_dtheta_valid_arr_1.at(2)) {
    vmask1 |= 0b1001;  // 14
  }
  if (best_dtheta_valid_arr_1.at(3)) {
    vmask2 |= 0b0110;  // 23
  }
  if (best_dtheta_valid_arr_1.at(4)) {
    vmask2 |= 0b1010;  // 24
  }
  if (best_dtheta_valid_arr_1.at(5)) {
    vmask3 |= 0b1100;  // 34
  }

  // merge station masks only if they share bits
  // Station 1 hits pass if any dTheta1X values pass
  // Station 2 hits pass if any dTheta2X values pass, *EXCEPT* the following cases:
  //           Only {13, 24} pass, only {13, 24, 34} pass,
  //           Only {14, 23} pass, only {14, 23, 34} pass.
  // Station 3 hits pass if any dTheta3X values pass, *EXCEPT* the following cases:
  //           Only {12, 34} pass, only {14, 23} pass.
  // Station 4 hits pass if any dTheta4X values pass, *EXCEPT* the following cases:
  //           Only {12, 34} pass, only {13, 24} pass.
  int vstat = vmask1;  // valid stations based on th coordinates
  if ((vstat & vmask2) != 0 || vstat == 0)
    vstat |= vmask2;
  if ((vstat & vmask3) != 0 || vstat == 0)
    vstat |= vmask3;

  // remove valid flag for station if hit does not pass the dTheta mask
  for (int istation = 0; istation < NUM_STATIONS; ++istation) {
    if ((vstat & (1 << istation)) == 0) {  // station bit not set
      st_conv_hits.at(istation).clear();
    }
  }

  // assign precise phi and theta for the track
  int phi_fp    = 0;
  int theta_fp  = 0;
  int best_pair = -1;

  if ((vstat & (1<<1)) != 0) {            // ME2 present
    if (!best_has_rpc_arr.at(0) && best_dtheta_valid_arr.at(0))      // 12, no RPC
      best_pair = 0;
    else if (!best_has_rpc_arr.at(3) && best_dtheta_valid_arr.at(3)) // 23, no RPC
      best_pair = 3;
    else if (!best_has_rpc_arr.at(4) && best_dtheta_valid_arr.at(4)) // 24, no RPC
      best_pair = 4;
    else if (best_dtheta_valid_arr.at(0)) // 12, has RPC
      best_pair = 0;
    else if (best_dtheta_valid_arr.at(3)) // 23, has RPC
      best_pair = 3;
    else if (best_dtheta_valid_arr.at(4)) // 24, has RPC
      best_pair = 4;
  } else if ((vstat & (1<<2)) != 0) {     // ME3 present
    if (!best_has_rpc_arr.at(1) && best_dtheta_valid_arr.at(1))      // 13, no RPC
      best_pair = 1;
    else if (!best_has_rpc_arr.at(5) && best_dtheta_valid_arr.at(5)) // 34, no RPC
      best_pair = 5;
    else if (best_dtheta_valid_arr.at(1)) // 13, has RPC
      best_pair = 1;
    else if (best_dtheta_valid_arr.at(5)) // 34, has RPC
      best_pair = 5;
  } else if ((vstat & (1<<3)) != 0) {     // ME4 present
    if (best_dtheta_valid_arr.at(2))      // 14
      best_pair = 2;
  }

  if (best_pair != -1) {
    phi_fp   = best_phi_arr.at(best_pair);
    theta_fp = best_theta_arr.at(best_pair);
    assert(theta_fp != 0);

    // In firmware, the track is associated to LCTs by the segment number, which
    // identifies the best strip, but does not resolve the ambiguity in theta.
    // In emulator, this additional logic also resolves the ambiguity in theta.
    struct {
      typedef EMTFHit value_type;
      bool operator()(const value_type& lhs, const value_type& rhs) const {
        return std::abs(lhs.Theta_fp()-theta) < std::abs(rhs.Theta_fp()-theta);
      }
      int theta;
    } less_dtheta_cmp;
    less_dtheta_cmp.theta = theta_fp;  // capture

    for (int istation = 0; istation < NUM_STATIONS; ++istation) {
      std::stable_sort(st_conv_hits.at(istation).begin(), st_conv_hits.at(istation).end(), less_dtheta_cmp);
      if (st_conv_hits.at(istation).size() > 1)
        st_conv_hits.at(istation).resize(1);  // just the minimum in dtheta
    }
  }

  // update rank taking into account available stations after theta deltas
  // keep straightness as it was
  int old_rank = (track.Rank() << 1);  // output rank is one bit longer than input rank, to accomodate ME4 separately
  int rank = (
      (((old_rank >> 6) & 1) << 6) |  // straightness
      (((old_rank >> 4) & 1) << 4) |  // straightness
      (((old_rank >> 2) & 1) << 2) |  // straightness
      (((vstat >> 0) & 1) << 5) |  // ME1
      (((vstat >> 1) & 1) << 3) |  // ME2
      (((vstat >> 2) & 1) << 1) |  // ME3
      (((vstat >> 3) & 1) << 0)    // ME4
  );

  int mode = (
      (((vstat >> 0) & 1) << 3) |  // ME1
      (((vstat >> 1) & 1) << 2) |  // ME2
      (((vstat >> 2) & 1) << 1) |  // ME3
      (((vstat >> 3) & 1) << 0)    // ME4
  );

  int mode_inv = vstat;

  // if less than 2 segments, kill rank
  if (vstat == 0b0001 || vstat == 0b0010 || vstat == 0b0100 || vstat == 0b1000 || vstat == 0)
    rank = 0;

  // From RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.cc
  auto isFront = [](int station, int ring, int chamber, int subsystem) {

    // // RPCs are behind the CSCs in stations 1, 3, and 4; in front in 2
    // if (subsystem == TriggerPrimitive::kRPC)
    //   return (station == 2);

    // In EMTF firmware, RPC hits are treated as if they came from the corresponding 
    // CSC chamber, so the FR bit assignment is the same as for CSCs - AWB 06.06.17

    // GEMs are in front of the CSCs
    if (subsystem == TriggerPrimitive::kGEM)
      return true;

    bool result = false;
    bool isOverlapping = !(station == 1 && ring == 3);
    // not overlapping means back
    if(isOverlapping)
    {
      bool isEven = (chamber % 2 == 0);
      // odd chambers are bolted to the iron, which faces
      // forward in 1&2, backward in 3&4, so...
      result = (station < 3) ? isEven : !isEven;
    }
    return result;
  };

  // Fill ptlut_data
  EMTFPtLUT ptlut_data;
  for (int i = 0; i < NUM_STATION_PAIRS; ++i) {
    ptlut_data.delta_ph[i] = best_dphi_arr.at(i);
    ptlut_data.sign_ph[i]  = best_dphi_sign_arr.at(i);
    ptlut_data.delta_th[i] = best_dtheta_arr.at(i);
    ptlut_data.sign_th[i]  = best_dtheta_sign_arr.at(i);
  }

  for (int i = 0; i < NUM_STATIONS; ++i) {
    const auto& v = st_conv_hits.at(i);
    ptlut_data.cpattern[i] = v.empty() ? 0 : v.front().Pattern();  // Automatically set to 0 for RPCs
    ptlut_data.fr[i]       = v.empty() ? 0 : isFront(v.front().Station(), v.front().Ring(), v.front().Chamber(), v.front().Subsystem());
    if (i == 0) 
      ptlut_data.st1_ring2 = v.empty() ? 0 : (v.front().Station() == 1 && (v.front().Ring() == 2 || v.front().Ring() == 3));
  }

  for (int i = 0; i < NUM_STATIONS+1; ++i) {  // 'bt' arrays use 5-station convention
    ptlut_data.bt_vi[i] = 0;
    ptlut_data.bt_hi[i] = 0;
    ptlut_data.bt_ci[i] = 0;
    ptlut_data.bt_si[i] = 0;
  }

  for (int i = 0; i < NUM_STATIONS; ++i) {
    const auto& v = st_conv_hits.at(i);
    if (!v.empty()) {
      int bt_station = v.front().BT_station();
      assert(0 <= bt_station && bt_station <= 4);

      int bt_segment = v.front().BT_segment();
      ptlut_data.bt_vi[bt_station] = 1;
      ptlut_data.bt_hi[bt_station] = (bt_segment >> 5) & 0x3;
      ptlut_data.bt_ci[bt_station] = (bt_segment >> 1) & 0xf;
      ptlut_data.bt_si[bt_station] = (bt_segment >> 0) & 0x1;
    }
  }

  // ___________________________________________________________________________
  // Output

  track.set_rank     ( rank );
  track.set_mode     ( mode );
  track.set_mode_inv ( mode_inv );
  track.set_phi_fp   ( phi_fp );
  track.set_theta_fp ( theta_fp );
  track.set_PtLUT    ( ptlut_data );

  track.set_phi_loc  ( emtf::calc_phi_loc_deg(phi_fp) );
  track.set_phi_glob ( emtf::calc_phi_glob_deg(track.Phi_loc(), track.Sector()) );
  track.set_theta    ( emtf::calc_theta_deg_from_int(theta_fp) );
  track.set_eta      ( emtf::calc_eta_from_theta_deg(track.Theta(), track.Endcap()) );

  // Only keep the best segments
  track.clear_Hits();

  EMTFHitCollection tmp_hits = track.Hits();
  flatten_container(st_conv_hits, tmp_hits);
  track.set_Hits( tmp_hits );

}

void AngleCalculation::calculate_bx(EMTFTrack& track) const {
  const int delayBX = bxWindow_ - 1;
  assert(delayBX >= 0);
  std::vector<int> counter(delayBX+1, 0);

  for (const auto& conv_hit : track.Hits()) {
    for (int i = delayBX; i >= 0; i--) {
      if (conv_hit.BX() <= bx_ - i)
        counter.at(i) += 1;  // Count stubs delayed by i BX or more
    }
  }

  int first_bx = bx_ - delayBX;
  int second_bx = 99;
  for (int i = delayBX; i >= 0; i--) {
    if (counter.at(i) >= 2) { // If 2 or more stubs are delayed by i BX or more
      second_bx = bx_ - i; // if i == delayBX, analyze immediately
      break;
    }
  }
  assert(second_bx != 99);

  // ___________________________________________________________________________
  // Output

  track.set_first_bx  ( first_bx );
  track.set_second_bx ( second_bx );
}

void AngleCalculation::erase_tracks(EMTFTrackCollection& tracks) const {
  // Erase tracks with rank == 0
  // using erase-remove idiom
  struct {
    typedef EMTFTrack value_type;
    bool operator()(const value_type& x) const {
      return (x.Rank() == 0);
    }
  } rank_zero_pred;

  tracks.erase(std::remove_if(tracks.begin(), tracks.end(), rank_zero_pred), tracks.end());

  for (const auto& track : tracks) {
    assert(track.Hits().size() > 0);
    assert(track.Hits().size() <= NUM_STATIONS);
  }
}
