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

  if (conv_hits.size() == 0)
    return;

  if (!useSingleHits_)
    return;

  if (int(best_tracks.size()) >= maxTracks_)
    return;

  // New collection to contain single-hit tracks
  EMTFTrackCollection one_hit_trks;

  EMTFHitCollection::const_iterator conv_hits_it  = conv_hits.begin();
  EMTFHitCollection::const_iterator conv_hits_end = conv_hits.end();

  // Loop over all the hits in a given BX
  for (; conv_hits_it != conv_hits_end; ++conv_hits_it) {

    // Only consider CSC LCTs
    if (conv_hits_it->Is_CSC() != 1)
      continue;

    // Only consider hits in station 1
    if (conv_hits_it->Station() != 1)
      continue;

    // Check if a hit has already been used in a track
    bool already_used = false;

    EMTFTrackCollection::iterator best_tracks_it  = best_tracks.begin();
    EMTFTrackCollection::iterator best_tracks_end = best_tracks.end();

    // Loop over existing multi-hit tracks
    for (; best_tracks_it != best_tracks_end; ++best_tracks_it) {

      // Only consider tracks with a hit in station 1
      if (best_tracks_it->Mode() < 8)
        continue;

      assert(best_tracks_it->Hits().front().Station() == 1);

      // Check if hit in track is identical
      // Copied from PrimitiveMatching.cc
      // "Duplicate" hits (with same strip but different wire) are considered identical
      const EMTFHit& conv_hit_i = *conv_hits_it;
      const EMTFHit& conv_hit_j = best_tracks_it->Hits().front();

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
        //(conv_hit_i.Roll()       == conv_hit_j.Roll()) &&
        true
      ) {
        already_used = true;
        break;
      }
    } // End loop: for (; best_tracks_it != best_tracks_end; ++best_tracks_it)

    // Only use hits that have not been used in a track
    if (already_used)
      continue;

    int zone = -1;
    int zone_code = conv_hits_it->Zone_code();
    if      (zone_code & 0b1000) zone = 4;
    else if (zone_code & 0b0100) zone = 3;
    else if (zone_code & 0b0010) zone = 2;
    else if (zone_code & 0b0001) zone = 1;
    else {
      std::cout << "\n\n EMTF SingleHitTrack.cc - bizzare case where zone_code = " << zone_code << std::endl;
      assert(zone > 0);
    }

    // Set "mode" using CLCT bend
    int mode = -1;
    int CLCT = conv_hits_it->Pattern();
    if      (CLCT >= 8) mode = 8;
    else if (CLCT >= 6) mode = 4;
    else if (CLCT >= 4) mode = 2;
    else if (CLCT >= 2) mode = 1;
    else {
      std::cout << "\n\n EMTF SingleHitTrack.cc - bizzare case where CLCT = " << CLCT << std::endl;
      assert(mode > 0);
    }


    EMTFTrack new_trk;
    new_trk.push_Hit ( *conv_hits_it );

    EMTFPtLUT empty_LUT;
    new_trk.set_PtLUT ( empty_LUT );

    new_trk.set_endcap       ( conv_hits_it->Endcap()     );
    new_trk.set_sector       ( conv_hits_it->Sector()     );
    new_trk.set_sector_idx   ( conv_hits_it->Sector_idx() );
    new_trk.set_mode         ( mode );
    new_trk.set_mode_inv     ( 0 );
    new_trk.set_rank         ( 0b0100000 );  // Station 1 hit, straightness 0 (see "rank" in AngleCalculation.cc)
    new_trk.set_winner       ( maxTracks_ - 1 );  // Always set to the last / lowest track
    new_trk.set_bx           ( bx_ );
    new_trk.set_first_bx     ( bx_ );
    new_trk.set_second_bx    ( bx_ );
    new_trk.set_zone         ( zone );
    new_trk.set_ph_num       ( conv_hits_it->Zone_hit() );
    new_trk.set_ph_q         ( 0b010000 );  // Original "quality_code" from PatternRecognition.cc
    new_trk.set_theta_fp     ( conv_hits_it->Theta_fp() );
    new_trk.set_theta        ( conv_hits_it->Theta() );
    new_trk.set_eta          ( conv_hits_it->Eta() );
    new_trk.set_phi_fp       ( conv_hits_it->Phi_fp() );
    new_trk.set_phi_loc      ( conv_hits_it->Phi_loc() );
    new_trk.set_phi_glob     ( conv_hits_it->Phi_glob() );
    new_trk.set_track_num    ( maxTracks_ - 1 );

    one_hit_trks.push_back( new_trk );
    if (int(best_tracks.size()) + int(one_hit_trks.size()) >= maxTracks_)
      break;

  } // End loop over conv_hits

  best_tracks.insert(best_tracks.end(), one_hit_trks.begin(), one_hit_trks.end());
}

