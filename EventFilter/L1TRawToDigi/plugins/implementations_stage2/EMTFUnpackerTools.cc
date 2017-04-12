
#include "EMTFUnpackerTools.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      
      void ImportME( EMTFHit& _hit, const l1t::emtf::ME _ME, 
		     const int _endcap, const int _sector ) {
	
	_hit.set_endcap     ( (_endcap == 1) ? 1 : -1 );
	_hit.set_sector_idx ( (_endcap == 1) ? _sector - 1 : _sector + 5 );
	
	_hit.set_wire       ( _ME.Wire() );
	_hit.set_strip      ( _ME.Strip() );
	_hit.set_quality    ( _ME.Quality() );
	_hit.set_pattern    ( _ME.CLCT_pattern() );
	_hit.set_bend       ( (_ME.LR() == 1) ? 1 : -1 );
	_hit.set_valid      ( _ME.VP() );
	_hit.set_sync_err   ( _ME.SE() );
	_hit.set_bx         ( _ME.TBIN() - 3 );
	_hit.set_bc0        ( _ME.BC0() ); 
	_hit.set_is_CSC     ( true  );
	_hit.set_is_RPC     ( false );
	_hit.set_subsystem  ( 1 );
	
	// Station, CSC_ID, Sector, Subsector, Neighbor, Sector_idx, Ring, and Chamber filled in
	// EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockME.cc
	// "set_layer()" is not invoked, so Layer is not yet filled - AWB 21.04.16
	
      } // End ImportME
      
      
      void ImportRPC( EMTFHit& _hit, const l1t::emtf::RPC _RPC,
		      const int _endcap, const int _sector ) {

	_hit.set_endcap     ( (_endcap == 1) ? 1 : -1 );
	_hit.set_sector     ( _sector ); // Not accounting for neighbors - AWB 15.03.17
	_hit.set_sector_idx ( (_endcap == 1) ? _sector - 1 : _sector + 5 );
	
	_hit.set_phi_fp    ( _RPC.Phi() );
	_hit.set_theta_fp  ( _RPC.Theta() );
	_hit.set_bx        ( _RPC.TBIN() - 3 );
	_hit.set_valid     ( _RPC.VP() );
	_hit.set_bc0       ( _RPC.BC0() ); 
	_hit.set_is_CSC    ( false );
	_hit.set_is_RPC    ( true  );
	// _hit.set_subsystem ( ? );  // Don't remember what RPC is - AWB 07.03.17
	
	// Sector, Subsector, Neighbor, and Sector_idx should be filled in
	// EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockRPC.cc - AWB 07.03.17
	
	_hit.set_phi_loc  ( L1TMuonEndCap::calc_phi_loc_deg        ( _RPC.Phi() ) );
	_hit.set_phi_glob ( L1TMuonEndCap::calc_phi_glob_deg       ( _hit.Phi_loc(), _sector ) );
	_hit.set_theta    ( L1TMuonEndCap::calc_theta_deg_from_int ( _RPC.Theta() ) );
	_hit.set_eta      ( L1TMuonEndCap::calc_eta_from_theta_deg ( _hit.Theta(), _endcap ) );
	
      } // End ImportRPC
      
      
      void ImportSP( EMTFTrack& _track, const l1t::emtf::SP _SP, 
		     const int _endcap, const int _sector ) {

	_track.set_endcap     ( (_endcap == 1) ? 1 : -1 );
	_track.set_sector     ( _sector );
	_track.set_sector_idx ( (_endcap == 1) ? _sector - 1 : _sector + 5 );
	_track.set_mode       ( _SP.Mode() );
	_track.set_mode_inv   ( (((_SP.Mode() >> 0) & 1) << 3) |
				(((_SP.Mode() >> 1) & 1) << 2) |
				(((_SP.Mode() >> 2) & 1) << 1) |
				(((_SP.Mode() >> 3) & 1) << 0) );
	_track.set_charge     ( (_SP.C() == 1) ? -1 : 1 ); // uGMT uses opposite of physical charge (to match pdgID)
	_track.set_bx         ( _SP.TBIN() - 3 );     
	_track.set_phi_fp     ( _SP.Phi_full() );  
	_track.set_phi_loc    ( L1TMuonEndCap::calc_phi_loc_deg ( _SP.Phi_full() ) );
	_track.set_phi_glob   ( L1TMuonEndCap::calc_phi_glob_deg( _track.Phi_loc(), _sector) );
	_track.set_pt         ( (_SP.Pt_GMT() - 1) * 0.5 );

	_track.set_gmt_pt     ( _SP.Pt_GMT() );
	_track.set_gmt_phi    ( _SP.Phi_GMT() );
	_track.set_gmt_eta    ( _SP.Eta_GMT() );
	_track.set_gmt_quality( _SP.Quality_GMT() );
	_track.set_gmt_charge ( _SP.C() );
	_track.set_gmt_charge_valid( _SP.VC() );

	EMTFPtLUT _lut;
	_lut.address = _SP.Pt_LUT_addr();
	_track.set_PtLUT( _lut );
	
	// First_bx, Second_bx, Track_num, Has_neighbor, All_neighbor, and Hits should be filled in
	// EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockSP.cc - AWB 07.03.17
	
      } // End ImportSP


    } // End namespace emtf                                                                                                                           
  } // End namespace stage2                                                                                                                           
} // End namespace l1t                                                                                                                                
