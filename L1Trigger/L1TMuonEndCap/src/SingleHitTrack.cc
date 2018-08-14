#include "L1Trigger/L1TMuonEndCap/interface/SingleHitTrack.h"

void SingleHitTrack::configure(
    int verbose, int endcap, int sector, int bx,
    int maxTracks,
    bool useSingleHits
) {
  verbose_ = verbose;
  endcap_  = endcap;
  sector_  = sector;
  bx_      = bx;

  maxTracks_ = maxTracks;

  useSingleHits_ = useSingleHits;
}

void SingleHitTrack::process(
    const EMTFHitCollection& conv_hits,
    EMTFTrackCollection& best_tracks
) const {

  if (conv_hits.empty())
    return;

  if (!useSingleHits_)
    return;

  if (int(best_tracks.size()) >= maxTracks_)
    return;

  // New collection to contain single-hit tracks
  EMTFTrackCollection one_hit_trks;

  // Loop over [subsector, CSC ID] pairs in order: [2,3], [2,2], [2,1], [1,3], [1,2], [1,1]
  for (int sub_ID = 5; sub_ID >= 0; sub_ID--) {
    int subsector = 1 + (sub_ID / 3);
    int CSC_ID    = 1 + (sub_ID % 3);

    // Loop over all the hits in a given BX
    for (const auto & conv_hits_it : conv_hits) {

      // Require subsector and CSC ID to match
      if (conv_hits_it.Subsector() != subsector || conv_hits_it.CSC_ID() != CSC_ID)
        continue;

      // Only consider CSC LCTs
      if (conv_hits_it.Is_CSC() != 1)
        continue;

      // Only consider hits in station 1, ring 1
      if (conv_hits_it.Station() != 1 || (conv_hits_it.Ring() % 3) != 1)
        continue;

      // Only consider hits in the same sector (not neighbor hits)
      if ( (conv_hits_it.Endcap() == 1) != (endcap_ == 1) || conv_hits_it.Sector() != sector_ )
        continue;

      // Check if a hit has already been used in a track
      bool already_used = false;

      // Loop over existing multi-hit tracks
      for (const auto & best_tracks_it : best_tracks) {

        // Only consider tracks with a hit in station 1
        if (best_tracks_it.Mode() < 8)
          continue;

        // Check if hit in track is identical
        // "Duplicate" hits (with same strip but different wire) are considered identical
        // const EMTFHit& conv_hit_i = *conv_hits_it;
        const EMTFHit& conv_hit_j = best_tracks_it.Hits().front();

        if (
            (conv_hits_it.Subsystem()  == conv_hit_j.Subsystem()) &&
            (conv_hits_it.PC_station() == conv_hit_j.PC_station()) &&
            (conv_hits_it.PC_chamber() == conv_hit_j.PC_chamber()) &&
            ((conv_hits_it.Ring() % 3) == (conv_hit_j.Ring() % 3)) &&  // because of ME1/1
            (conv_hits_it.Strip()      == conv_hit_j.Strip()) &&
            // (conv_hits_it.Wire()       == conv_hit_j.Wire()) &&
            (conv_hits_it.BX()         == conv_hit_j.BX()) &&
            true
            ) {
          already_used = true;
          break;
        }
      } // End loop: for (const auto & best_tracks_it : best_tracks)

      // Only use hits that have not been used in a track
      if (already_used)
        continue;

      int zone = -1;
      int zone_code = conv_hits_it.Zone_code();
      if      (zone_code & 0b1000) zone = 4;
      else if (zone_code & 0b0100) zone = 3;
      else if (zone_code & 0b0010) zone = 2;
      else if (zone_code & 0b0001) zone = 1;
      else {
	edm::LogError("L1T") << "EMTF SingleHitTrack.cc - bizzare case where zone_code = " << zone_code; return;
      }

      EMTFTrack new_trk;
      new_trk.push_Hit ( conv_hits_it );

      EMTFPtLUT empty_LUT = {};
      new_trk.set_PtLUT ( empty_LUT );

      new_trk.set_endcap       ( conv_hits_it.Endcap()     );
      new_trk.set_sector       ( conv_hits_it.Sector()     );
      new_trk.set_sector_idx   ( conv_hits_it.Sector_idx() );
      new_trk.set_mode         ( 1 ); // Set "mode" to 1
      new_trk.set_mode_inv     ( 0 );
      new_trk.set_rank         ( 0b0100000 );  // Station 1 hit, straightness 0 (see "rank" in AngleCalculation.cc)
      new_trk.set_winner       ( maxTracks_ - 1 );  // Always set to the last / lowest track
      new_trk.set_bx           ( bx_ );
      new_trk.set_first_bx     ( bx_ );
      new_trk.set_second_bx    ( bx_ );
      new_trk.set_zone         ( zone );
      new_trk.set_ph_num       ( conv_hits_it.Zone_hit() );
      new_trk.set_ph_q         ( 0b010000 );  // Original "quality_code" from PatternRecognition.cc
      new_trk.set_theta_fp     ( conv_hits_it.Theta_fp() );
      new_trk.set_theta        ( conv_hits_it.Theta() );
      new_trk.set_eta          ( conv_hits_it.Eta() );
      new_trk.set_phi_fp       ( conv_hits_it.Phi_fp() );
      new_trk.set_phi_loc      ( conv_hits_it.Phi_loc() );
      new_trk.set_phi_glob     ( conv_hits_it.Phi_glob() );
      new_trk.set_track_num    ( maxTracks_ - 1 );

      one_hit_trks.push_back( new_trk );

      if (int(best_tracks.size()) + int(one_hit_trks.size()) >= maxTracks_)
        break;

      // Firmware only sends one single-hit track per sector
      if (!one_hit_trks.empty())
        break;

    } // End loop:  for (const auto & conv_hits_it : conv_hits)

    if (!one_hit_trks.empty())
      break;

  } // End loop: for (int sub_ID = 5; sub_ID > 0; sub_ID--) {


  best_tracks.insert(best_tracks.end(), one_hit_trks.begin(), one_hit_trks.end());
}

