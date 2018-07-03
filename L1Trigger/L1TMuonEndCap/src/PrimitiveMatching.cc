#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveMatching.h"

#include "helper.h"  // to_hex, to_binary, merge_sort3

namespace {
  const int bw_fph = 13;  // bit width of ph, full precision
  const int bpow = 7;     // (1 << bpow) is count of input ranks
  const int invalid_ph_diff = 0x1ff;  // 511 (9-bit)
}


void PrimitiveMatching::configure(
    int verbose, int endcap, int sector, int bx,
    bool fixZonePhi, bool useNewZones,
    bool bugSt2PhDiff, bool bugME11Dupes
) {
  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  fixZonePhi_      = fixZonePhi;
  useNewZones_     = useNewZones;
  bugSt2PhDiff_    = bugSt2PhDiff;
  bugME11Dupes_    = bugME11Dupes;
}

void PrimitiveMatching::process(
    const std::deque<EMTFHitCollection>& extended_conv_hits,
    const emtf::zone_array<EMTFRoadCollection>& zone_roads,
    emtf::zone_array<EMTFTrackCollection>& zone_tracks
) const {

  // Function to update fs_history encoded in fs_segment
  auto update_fs_history = [](int fs_segment, int this_bx, int hit_bx) {
    // 0 for current BX, 1 for previous BX, 2 for BX before that
    int fs_history = this_bx - hit_bx;
    fs_segment |= ((fs_history & 0x3)<<4);
    return fs_segment;
  };

  // Function to update bt_history encoded in bt_segment
  auto update_bt_history = [](int bt_segment, int this_bx, int hit_bx) {
    // 0 for current BX, 1 for previous BX, 2 for BX before that
    int bt_history = this_bx - hit_bx;
    bt_segment |= ((bt_history & 0x3)<<5);
    return bt_segment;
  };


  // Exit if no roads
  int num_roads = 0;
  for (const auto& roads : zone_roads)
    num_roads += roads.size();
  bool early_exit = (num_roads == 0);

  if (early_exit)
    return;


  if (verbose_ > 0) {  // debug
    for (const auto& roads : zone_roads) {
      for (const auto& road : roads) {
        std::cout << "pattern on match input: z: " << road.Zone()-1 << " r: " << road.Winner()
            << " ph_num: " << road.Key_zhit() << " ph_q: " << to_hex(road.Quality_code())
            << " ly: " << to_binary(road.Layer_code(), 3) << " str: " << to_binary(road.Straightness(), 3)
            << std::endl;
      }
    }
  }

  // Organize converted hits by (zone, station)
  std::array<EMTFHitCollection, emtf::NUM_ZONES*emtf::NUM_STATIONS> zs_conv_hits;

  bool use_fs_zone_code = true;  // use zone code as in firmware find_segment module

  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_it  = extended_conv_hits.begin();
  std::deque<EMTFHitCollection>::const_iterator ext_conv_hits_end = extended_conv_hits.end();

  for (; ext_conv_hits_it != ext_conv_hits_end; ++ext_conv_hits_it) {
    EMTFHitCollection::const_iterator conv_hits_it  = ext_conv_hits_it->begin();
    EMTFHitCollection::const_iterator conv_hits_end = ext_conv_hits_it->end();

    for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {
      int istation = conv_hits_it->Station()-1;
      int zone_code = conv_hits_it->Zone_code();  // decide based on original zone code
      if (use_fs_zone_code)
        zone_code = conv_hits_it->FS_zone_code();  // decide based on new zone code

      // A hit can go into multiple zones
      for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
        if (!zone_roads.at(izone).empty()) {

          if (zone_code & (1<<izone)) {
            const int zs = (izone*emtf::NUM_STATIONS) + istation;
            zs_conv_hits.at(zs).push_back(*conv_hits_it);

            // Update fs_history and bt_history depending on the processor BX
            // This update only goes into the hits associated to a track, it does not affect the original hit collection
            EMTFHit& conv_hit = zs_conv_hits.at(zs).back();   // pass by reference
            int old_fs_segment = conv_hit.FS_segment();
            int new_fs_segment = update_fs_history(old_fs_segment, bx_, conv_hit.BX());
            conv_hit.set_fs_segment( new_fs_segment );

            int old_bt_segment = conv_hit.BT_segment();
            int new_bt_segment = update_bt_history(old_bt_segment, bx_, conv_hit.BX());
            conv_hit.set_bt_segment( new_bt_segment );
          }
        }
      }

    }  // end loop over conv_hits
  }  // end loop over extended_conv_hits

  if (verbose_ > 1) {  // debug
    for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
      for (int istation = 0; istation < emtf::NUM_STATIONS; ++istation) {
        const int zs = (izone*emtf::NUM_STATIONS) + istation;
        for (const auto& conv_hit : zs_conv_hits.at(zs)) {
          std::cout << "z: " << izone << " st: " << istation+1 << " cscid: " << conv_hit.CSC_ID()
              << " ph_zone_phi: " << conv_hit.Zone_hit() << " ph_low_prec: " << (conv_hit.Zone_hit()<<5)
              << " ph_high_prec: " << conv_hit.Phi_fp() << " ph_high_low_diff: " << (conv_hit.Phi_fp() - (conv_hit.Zone_hit()<<5))
              << std::endl;
        }
      }
    }
  }

  // Keep the best phi difference for every road by (zone, station)
  std::array<std::vector<hit_sort_pair_t>, emtf::NUM_ZONES*emtf::NUM_STATIONS> zs_phi_differences;

  // Get the best-matching hits by comparing phi difference between
  // pattern and segment
  for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
    for (int istation = 0; istation < emtf::NUM_STATIONS; ++istation) {
      const int zs = (izone*emtf::NUM_STATIONS) + istation;

      // This leaves zone_roads.at(izone) and zs_conv_hits.at(zs) unchanged
      // zs_phi_differences.at(zs) gets filled with a pair of <phi_diff, conv_hit> for the
      // conv_hit with the lowest phi_diff from the pattern in this station and zone
      process_single_zone_station(
          izone+1, istation+1,
          zone_roads.at(izone),
          zs_conv_hits.at(zs),
          zs_phi_differences.at(zs)
      );

      if (not(zone_roads.at(izone).size() == zs_phi_differences.at(zs).size()))
	{ edm::LogError("L1T") << "zone_roads.at(izone).size() = " << zone_roads.at(izone).size()
			       << ", zs_phi_differences.at(zs).size() = " << zs_phi_differences.at(zs).size(); return; }
    }  // end loop over stations
  }  // end loop over zones

  if (verbose_ > 1) {  // debug
    for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
      const auto& roads = zone_roads.at(izone);
      for (unsigned iroad = 0; iroad < roads.size(); ++iroad) {
        const auto& road = roads.at(iroad);
        for (int istation = 0; istation < emtf::NUM_STATIONS; ++istation) {
          const int zs = (izone*emtf::NUM_STATIONS) + istation;
          int ph_diff = zs_phi_differences.at(zs).at(iroad).first;
          std::cout << "find seg: z: " << road.Zone()-1 << " r: " << road.Winner()
              << " st: " << istation << " ph_diff: " << ph_diff
              << std::endl;
        }
      }
    }
  }


  // Build all tracks in each zone
  for (int izone = 0; izone < emtf::NUM_ZONES; ++izone) {
    const EMTFRoadCollection& roads = zone_roads.at(izone);

    for (unsigned iroad = 0; iroad < roads.size(); ++iroad) {
      const EMTFRoad& road = roads.at(iroad);

      // Create a track
      EMTFTrack track;
      track.set_endcap   ( road.Endcap() );
      track.set_sector   ( road.Sector() );
      track.set_sector_idx ( road.Sector_idx() );
      track.set_bx       ( road.BX() );
      track.set_zone     ( road.Zone() );
      track.set_ph_num   ( road.Key_zhit() );
      track.set_ph_q     ( road.Quality_code() );
      track.set_rank     ( road.Quality_code() );
      track.set_winner   ( road.Winner() );

      track.clear_Hits();

      // Insert hits
      for (int istation = 0; istation < emtf::NUM_STATIONS; ++istation) {
        const int zs = (izone*emtf::NUM_STATIONS) + istation;

        const EMTFHitCollection& conv_hits = zs_conv_hits.at(zs);
        int       ph_diff      = zs_phi_differences.at(zs).at(iroad).first;
        hit_ptr_t conv_hit_ptr = zs_phi_differences.at(zs).at(iroad).second;

        if (ph_diff != invalid_ph_diff) {
          // Inserts the conv_hit with the lowest phi_diff, as well as its duplicate
          // (same strip and phi, different wire and theta), if a duplicate exists
          insert_hits(conv_hit_ptr, conv_hits, track);
        }
      }

      if (fixZonePhi_) {
        if (not(!track.Hits().empty()))
	  { edm::LogError("L1T") << "track.Hits().empty() = " << track.Hits().empty(); return; }
      }

      // Output track
      zone_tracks.at(izone).push_back(track);

    }  // end loop over roads
  }  // end loop over zones

  if (verbose_ > 0) {  // debug
    for (const auto& tracks : zone_tracks) {
      for (const auto& track : tracks) {
        for (const auto& hit : track.Hits()) {
          std::cout << "match seg: z: " << track.Zone()-1 << " pat: " << track.Winner() <<  " st: " << hit.Station()
              << " vi: " << to_binary(0b1, 2) << " hi: " << ((hit.FS_segment()>>4) & 0x3)
              << " ci: " << ((hit.FS_segment()>>1) & 0x7) << " si: " << (hit.FS_segment() & 0x1)
              << " ph: " << hit.Phi_fp() << " th: " << hit.Theta_fp()
              << std::endl;
        }
      }
    }
  }

}

void PrimitiveMatching::process_single_zone_station(
    int zone, int station,
    const EMTFRoadCollection& roads,
    const EMTFHitCollection& conv_hits,
    std::vector<hit_sort_pair_t>& phi_differences
) const {
  // max phi difference between pattern and segment
  // This doesn't depend on the pattern straightness - any hit within the max phi difference may match
  int max_ph_diff = (station == 1) ? 15 : 7;
  //int bw_ph_diff = (station == 1) ? 5 : 4; // ph difference bit width
  //int invalid_ph_diff = (station == 1) ? 31 : 15;  // invalid difference

  if (fixZonePhi_) {
    if (station == 1) {
      max_ph_diff = 496;  // width of pattern in ME1 + rounding error 15*32+16
      //bw_ph_diff = 9;
      //invalid_ph_diff = 0x1ff;
    } else if (station == 2) {
      if (bugSt2PhDiff_)
        max_ph_diff = 16;   // just rounding error for ME2 (pattern must match ME2 hit phi if there was one)
      else
        max_ph_diff = 240;  // same as ME3,4
      //bw_ph_diff = 5;
      //invalid_ph_diff = 0x1f;
    } else {
      max_ph_diff = 240;  // width of pattern in ME3,4 + rounding error 7*32+16
      //bw_ph_diff = 8;
      //invalid_ph_diff = 0xff;
    }
  }

  auto abs_diff = [](int a, int b) { return std::abs(a-b); };

  // Simple sort by ph_diff
  struct {
    typedef hit_sort_pair_t value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      return lhs.first <= rhs.first;
    }
  } less_ph_diff_cmp;

  // Emulation of FW sorting with 3-way comparator
  struct {
    typedef hit_sort_pair_t value_type;
    int operator()(const value_type& a, const value_type& b, const value_type& c) const {
      int r = 0;
      r |= bool(a.first <= b.first);
      r <<= 1;
      r |= bool(b.first <= c.first);
      r <<= 1;
      r |= bool(c.first <= a.first);

      int rr = 0;
      switch(r) {
      //case 0b000 : rr = 3; break;  // invalid
      case 0b001 : rr = 2; break;  // c
      case 0b010 : rr = 1; break;  // b
      case 0b011 : rr = 1; break;  // b
      case 0b100 : rr = 0; break;  // a
      case 0b101 : rr = 2; break;  // c
      case 0b110 : rr = 0; break;  // a
      //case 0b111 : rr = 0; break;  // invalid
      default    : rr = 0; break;
      }
      return rr;
    }
  } less_ph_diff_cmp3;


  // ___________________________________________________________________________
  // For each road, find the segment with min phi difference in every station

  EMTFRoadCollection::const_iterator roads_it  = roads.begin();
  EMTFRoadCollection::const_iterator roads_end = roads.end();

  for (; roads_it != roads_end; ++roads_it) {
    int ph_pat = roads_it->Key_zhit();     // pattern key phi value
    int ph_q   = roads_it->Quality_code(); // pattern quality code
    if (not(ph_pat >= 0 && ph_q > 0))
      { edm::LogError("L1T") << "ph_pat = " << ph_pat << ", ph_q = " << ph_q; return; }

    if (fixZonePhi_) {
      ph_pat <<= 5;  // add missing 5 lower bits to pattern phi
    }

    std::vector<hit_sort_pair_t> tmp_phi_differences;

    EMTFHitCollection::const_iterator conv_hits_it  = conv_hits.begin();
    EMTFHitCollection::const_iterator conv_hits_end = conv_hits.end();

    for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {
      int ph_seg     = conv_hits_it->Phi_fp();  // ph from segments
      int ph_seg_red = ph_seg >> (bw_fph-bpow-1);  // remove unused low bits
      if (not(ph_seg >= 0))
	{ edm::LogError("L1T") << "ph_seg = " << ph_seg; return; }

      if (fixZonePhi_) {
        ph_seg_red = ph_seg;  // use full-precision phi
      }

      // Get abs phi difference
      int ph_diff = abs_diff(ph_pat, ph_seg_red);
      if (ph_diff > max_ph_diff)
        ph_diff = invalid_ph_diff;  // difference is too high, cannot be the same pattern

      if (ph_diff != invalid_ph_diff)
        tmp_phi_differences.push_back(std::make_pair(ph_diff, conv_hits_it));  // make a key-value pair
    }

    // _________________________________________________________________________
    // Sort to find the segment with min phi difference

    if (!tmp_phi_differences.empty()) {
      // Because the sorting is sensitive to FW ordering, use the exact FW sorting.
      // This implementation still slightly differs from FW because I prefer to
      // use a sorting function that is as generic as possible.
      bool use_fw_sorting = true;

      if (useNewZones_)  use_fw_sorting = false;

      if (use_fw_sorting) {
        // zone_cham = 4 for [fs_01, fs_02, fs_03, fs_11], or 7 otherwise
        // tot_diff = 27 or 45 in FW; it is 27 or 54 in the C++ merge_sort3 impl
        const int max_drift = 3; // should use bxWindow from the config
        const int zone_cham = ((zone == 1 && (2 <= station && station <= 4)) || (zone == 2 && station == 2)) ? 4 : 7;
        const int seg_ch    = 2;
        const int tot_diff  = (max_drift*zone_cham*seg_ch) + ((zone_cham == 4) ? 3 : 12);  // provide padding for 3-input comparators

        std::vector<hit_sort_pair_t> fw_sort_array(tot_diff, std::make_pair(invalid_ph_diff, conv_hits_end));

        // FW doesn't check if the hit is CSC or RPC
        std::vector<hit_sort_pair_t>::const_iterator phdiffs_it  = tmp_phi_differences.begin();
        std::vector<hit_sort_pair_t>::const_iterator phdiffs_end = tmp_phi_differences.end();

        for (; phdiffs_it != phdiffs_end; ++phdiffs_it) {
          //int ph_diff    = phdiffs_it->first;
          int fs_segment = phdiffs_it->second->FS_segment();

          // Calculate the index to put into the fw_sort_array
          int fs_history = ((fs_segment>>4) & 0x3);
          int fs_chamber = ((fs_segment>>1) & 0x7);
          fs_segment = (fs_segment & 0x1);
          unsigned fw_sort_array_index = (fs_history * zone_cham * seg_ch) + (fs_chamber * seg_ch) + fs_segment;

          if (not(fs_history < max_drift && fs_chamber < zone_cham && fs_segment < seg_ch))
	    { edm::LogError("L1T") << "fs_history = " << fs_history << ", max_drift = " << max_drift
				   << ", fs_chamber = " << fs_chamber << ", zone_cham = " << zone_cham
				   << ", fs_segment = " << fs_segment << ", seg_ch = " << seg_ch; return; }
          if (not(fw_sort_array_index < fw_sort_array.size()))
	    { edm::LogError("L1T") << "fw_sort_array_index = " << fw_sort_array_index
				   << ", fw_sort_array.size() = " << fw_sort_array.size(); return; }
          fw_sort_array.at(fw_sort_array_index) = *phdiffs_it;
        }

        // Debug
        //std::cout << "phdiffs" << std::endl;
        //for (unsigned i = 0; i < fw_sort_array.size(); ++i)
        //  std::cout << fw_sort_array.at(i).first << " ";
        //std::cout << std::endl;

        // Debug
        //std::cout << "Before sort" << std::endl;
        //for (unsigned i = 0; i < fw_sort_array.size(); ++i)
        //  std::cout << fw_sort_array.at(i).second->FS_segment() << " ";
        //std::cout << std::endl;

        // Find the best phi difference according to FW sorting
        //merge_sort3(fw_sort_array.begin(), fw_sort_array.end(), less_ph_diff_cmp, less_ph_diff_cmp3);
        merge_sort3_with_hint(fw_sort_array.begin(), fw_sort_array.end(), less_ph_diff_cmp, less_ph_diff_cmp3, ((tot_diff == 54) ? tot_diff/2 : tot_diff/3));

        // Store the best phi difference
        phi_differences.push_back(fw_sort_array.front());

        // Debug
        //std::cout << "After sort" << std::endl;
        //for (unsigned i = 0; i < fw_sort_array.size(); ++i)
        //  std::cout << fw_sort_array.at(i).second->FS_segment() << " ";
        //std::cout << std::endl;

      } else {  // use C++ sorting
        struct {
          typedef hit_sort_pair_t value_type;
          bool operator()(const value_type& lhs, const value_type& rhs) const {
            // If different types, prefer CSC over RPC; else prefer the closer hit in dPhi
            if (lhs.second->Subsystem() != rhs.second->Subsystem())
              return (lhs.second->Subsystem() == TriggerPrimitive::kCSC);
            else
              return lhs.first <= rhs.first;
          }
        } tmp_less_ph_diff_cmp;

        // Find best phi difference
        std::stable_sort(tmp_phi_differences.begin(), tmp_phi_differences.end(), tmp_less_ph_diff_cmp);

        // Store the best phi difference
        phi_differences.push_back(tmp_phi_differences.front());
      }

    } else {
      // No segment found
      phi_differences.push_back(std::make_pair(invalid_ph_diff, conv_hits_end));  // make a key-value pair
    }

  }  // end loop over roads
}

void PrimitiveMatching::insert_hits(
    hit_ptr_t conv_hit_ptr, const EMTFHitCollection& conv_hits,
    EMTFTrack& track
) const {
  EMTFHitCollection::const_iterator conv_hits_it  = conv_hits.begin();
  EMTFHitCollection::const_iterator conv_hits_end = conv_hits.end();

  const bool is_csc_me11 = (conv_hit_ptr->Subsystem() == TriggerPrimitive::kCSC) &&
      (conv_hit_ptr->Station() == 1) && (conv_hit_ptr->Ring() == 1 || conv_hit_ptr->Ring() == 4);

  // Find all possible duplicated hits, insert them
  for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {
    const EMTFHit& conv_hit_i = *conv_hits_it;
    const EMTFHit& conv_hit_j = *conv_hit_ptr;

    // All these must match: [bx_history][station][chamber][segment]
    if (
      (conv_hit_i.Subsystem()  == conv_hit_j.Subsystem()) &&
      (conv_hit_i.PC_station() == conv_hit_j.PC_station()) &&
      (conv_hit_i.PC_chamber() == conv_hit_j.PC_chamber()) &&
      (conv_hit_i.Ring()       == conv_hit_j.Ring()) &&  // because of ME1/1
      (conv_hit_i.Strip()      == conv_hit_j.Strip()) &&
      //(conv_hit_i.Wire()       == conv_hit_j.Wire()) &&
      (conv_hit_i.Pattern()    == conv_hit_j.Pattern()) &&
      (conv_hit_i.BX()         == conv_hit_j.BX()) &&
      (conv_hit_i.Strip_low()  == conv_hit_j.Strip_low()) && // For RPC clusters
      (conv_hit_i.Strip_hi()   == conv_hit_j.Strip_hi()) &&  // For RPC clusters
      (conv_hit_i.Roll()       == conv_hit_j.Roll()) &&      // For RPC clusters
      true
    ) {
      // All duplicates with the same strip but different wire must have same phi_fp
      if (not(conv_hit_i.Phi_fp() == conv_hit_j.Phi_fp()))
	{ edm::LogError("L1T") << "conv_hit_i.Phi_fp() = " << conv_hit_i.Phi_fp() 
			       << ", conv_hit_j.Phi_fp() = " << conv_hit_j.Phi_fp(); return; }

      track.push_Hit( conv_hit_i );

    } else if (
      (bugME11Dupes_ && is_csc_me11) &&  // if reproduce ME1/1 theta duplication bug, do not check 'ring', 'strip' and 'pattern'
      (conv_hit_i.Subsystem()  == conv_hit_j.Subsystem()) &&
      (conv_hit_i.PC_station() == conv_hit_j.PC_station()) &&
      (conv_hit_i.PC_chamber() == conv_hit_j.PC_chamber()) &&
      //(conv_hit_i.Ring()       == conv_hit_j.Ring()) &&  // because of ME1/1
      //(conv_hit_i.Strip()      == conv_hit_j.Strip()) &&
      //(conv_hit_i.Wire()       == conv_hit_j.Wire()) &&
      //(conv_hit_i.Pattern()    == conv_hit_j.Pattern()) &&
      (conv_hit_i.BX()         == conv_hit_j.BX()) &&
      //(conv_hit_i.Strip_low()  == conv_hit_j.Strip_low()) && // For RPC clusters
      //(conv_hit_i.Strip_hi()   == conv_hit_j.Strip_hi()) &&  // For RPC clusters
      //(conv_hit_i.Roll()       == conv_hit_j.Roll()) &&      // For RPC clusters
      true
    ) {
      // Dirty hack
      EMTFHit tmp_hit = conv_hit_j;
      tmp_hit.set_theta_fp( conv_hit_i.Theta_fp() );
      track.push_Hit( tmp_hit );
    }
  }

  // Sort by station
  struct {
    typedef EMTFHit value_type;
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      return lhs.Station() < rhs.Station();
    }
  } less_station_cmp;

  EMTFHitCollection tmp_hits = track.Hits();
  std::stable_sort(tmp_hits.begin(), tmp_hits.end(), less_station_cmp);
  track.set_Hits( tmp_hits );
}
