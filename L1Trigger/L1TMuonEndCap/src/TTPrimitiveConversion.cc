#include "L1Trigger/L1TMuonEndCap/interface/TTPrimitiveConversion.h"

#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.h"
#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"


void TTPrimitiveConversion::configure(
    const TTGeometryTranslator* tp_ttgeom,
    const SectorProcessorLUT* lut,
    int verbose, int endcap, int sector, int bx
) {
  assert(tp_ttgeom != nullptr);
  assert(lut != nullptr);

  tp_ttgeom_ = tp_ttgeom;
  lut_       = lut;  // not used

  verbose_ = verbose;
  endcap_  = endcap; // 1 for ME+, 2 for ME-
  sector_  = sector;
  bx_      = bx;
}

void TTPrimitiveConversion::process(
    const std::map<int, TTTriggerPrimitiveCollection>& selected_ttprim_map,
    EMTFHitCollection& conv_hits
) const {

  for (const auto& map_tp_it : selected_ttprim_map) {
    for (const auto& tp_it : map_tp_it.second) {
      EMTFHit conv_hit;
      convert_tt(tp_it, conv_hit);
      conv_hits.push_back(conv_hit);
    }
  }
}

void TTPrimitiveConversion::process_no_prim_sel(
    const TTTriggerPrimitiveCollection& ttmuon_primitives,
    EMTFHitCollection& conv_hits
) const {

  for (const auto& tp_it : ttmuon_primitives) {
    if (endcap_ == 1 && sector_ == 1 && bx_ == tp_it.getTTData().bx) {  //FIXME: stupidly put everything into sector +1, to be fixed.
      EMTFHit conv_hit;
      convert_tt(tp_it, conv_hit);
      conv_hits.push_back(conv_hit);
    }
  }
}


// _____________________________________________________________________________
// TT functions
void TTPrimitiveConversion::convert_tt(
    const TTTriggerPrimitive& ttmuon_primitive,
    EMTFHit& conv_hit
) const {
  //const DetId&   tp_detId = ttmuon_primitive.detId();
  const TTData&  tp_data  = ttmuon_primitive.getTTData();

  int tp_region    = tp_ttgeom_->region(ttmuon_primitive);     // 0 for Barrel, +/-1 for +/- Endcap
  int tp_endcap    = (tp_region == -1) ? 2 : tp_region;
  int tp_station   = tp_ttgeom_->layer(ttmuon_primitive);
  int tp_ring      = tp_ttgeom_->ring(ttmuon_primitive);
  int tp_chamber   = tp_ttgeom_->module(ttmuon_primitive);
  int tp_sector    = 1;  //FIXME
  int tp_subsector = 0;  //FIXME

  const bool is_neighbor = false;  //FIXME

  // Set properties
  //conv_hit.SetTTDetId      ( tp_detId );

  conv_hit.set_endcap      ( (tp_endcap == 2) ? -1 : tp_endcap );
  conv_hit.set_station     ( tp_station );
  conv_hit.set_ring        ( tp_ring );
  //conv_hit.set_roll        ( tp_roll );
  conv_hit.set_chamber     ( tp_chamber );
  conv_hit.set_sector      ( tp_sector );
  conv_hit.set_subsector   ( tp_subsector );
  //conv_hit.set_csc_ID      ( tp_csc_ID );
  //conv_hit.set_csc_nID     ( csc_nID );
  //conv_hit.set_track_num   ( tp_data.trknmb );
  //conv_hit.set_sync_err    ( tp_data.syncErr );

  conv_hit.set_bx          ( tp_data.bx );
  conv_hit.set_subsystem   ( TTTriggerPrimitive::kTT );
  conv_hit.set_is_CSC      ( false );
  conv_hit.set_is_RPC      ( false );
  conv_hit.set_is_GEM      ( false );

  //conv_hit.set_pc_sector   ( pc_sector );
  //conv_hit.set_pc_station  ( pc_station );
  //conv_hit.set_pc_chamber  ( pc_chamber );
  //conv_hit.set_pc_segment  ( pc_segment );

  conv_hit.set_valid       ( true );
  conv_hit.set_strip       ( static_cast<int>(tp_data.row_f) );
  //conv_hit.set_strip_low   ( tp_data.strip_low );
  //conv_hit.set_strip_hi    ( tp_data.strip_hi );
  conv_hit.set_wire        ( static_cast<int>(tp_data.col_f) );
  //conv_hit.set_quality     ( tp_data.quality );
  //conv_hit.set_pattern     ( tp_data.pattern );
  conv_hit.set_bend        ( tp_data.bend );
  //conv_hit.set_time        ( tp_data.time );

  conv_hit.set_neighbor    ( is_neighbor );
  conv_hit.set_sector_idx  ( (endcap_ == 1) ? sector_ - 1 : sector_ + 5 );

  // Add coordinates from fullsim
  {
    const GlobalPoint& gp = tp_ttgeom_->getGlobalPoint(ttmuon_primitive);
    double glob_phi   = emtf::rad_to_deg(gp.phi().value());
    double glob_theta = emtf::rad_to_deg(gp.theta());
    double glob_eta   = gp.eta();
    double glob_rho   = gp.perp();
    double glob_z     = gp.z();

    conv_hit.set_phi_sim   ( glob_phi );
    conv_hit.set_theta_sim ( glob_theta );
    conv_hit.set_eta_sim   ( glob_eta );
    conv_hit.set_rho_sim   ( glob_rho );
    conv_hit.set_z_sim     ( glob_z );
  }
}
