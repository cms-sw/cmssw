
#include "L1Trigger/L1TMuonEndCap/interface/EMTFTrackTools.h"

namespace l1t {

  void EMTFTrack::ImportSP( const emtf::SP _SP, int _sector) {

    EMTFTrack::set_sector       ( _sector );
    EMTFTrack::set_sector_GMT   ( calc_sector_GMT(_sector) );
    EMTFTrack::set_mode         ( _SP.Mode() );         
    EMTFTrack::set_quality      ( _SP.Quality_GMT() );      
    EMTFTrack::set_bx           ( _SP.TBIN() - 3 );     
    EMTFTrack::set_pt_GMT       ( _SP.Pt_GMT() );       
    EMTFTrack::set_pt_LUT_addr  ( _SP.Pt_LUT_addr() );
    EMTFTrack::set_eta_GMT      ( _SP.Eta_GMT() );      
    EMTFTrack::set_phi_loc_int  ( _SP.Phi_full() );  
    EMTFTrack::set_phi_GMT      ( _SP.Phi_GMT() );      
    EMTFTrack::set_charge       ( (_SP.C() == 1) ? -1 : 1 ); // uGMT uses opposite of physical charge (to match pdgID)
    EMTFTrack::set_charge_GMT   ( _SP.C() );
    EMTFTrack::set_charge_valid ( _SP.VC() );

    EMTFTrack::set_pt           ( calc_pt( pt_GMT ) );
    EMTFTrack::set_eta          ( calc_eta( eta_GMT ) );
    EMTFTrack::set_phi_loc_deg  ( calc_phi_loc_deg( phi_loc_int ) );
    EMTFTrack::set_phi_loc_rad  ( calc_phi_loc_rad( phi_loc_int ) );
    EMTFTrack::set_phi_glob_deg ( calc_phi_glob_deg( phi_loc_deg, _sector ) );
    EMTFTrack::set_phi_glob_rad ( calc_phi_glob_rad( phi_loc_rad, _sector ) );

  } // End EMTFTrack::ImportSP

  // Calculates special chamber ID for track address sent to uGMT, using CSC_ID, subsector, neighbor, and station
  int calc_uGMT_chamber( int _csc_ID, int _subsector, int _neighbor, int _station) {
    if (_station == 1) {
      if      ( _csc_ID == 3 && _neighbor == 1 && _subsector == 2 ) return 1;
      else if ( _csc_ID == 6 && _neighbor == 1 && _subsector == 2 ) return 2;
      else if ( _csc_ID == 9 && _neighbor == 1 && _subsector == 2 ) return 3;
      else if ( _csc_ID == 3 && _neighbor == 0 && _subsector == 2 ) return 4;
      else if ( _csc_ID == 6 && _neighbor == 0 && _subsector == 2 ) return 5;
      else if ( _csc_ID == 9 && _neighbor == 0 && _subsector == 2 ) return 6;
      else return 0;
    }
    else {
      if      ( _csc_ID == 3 && _neighbor == 1 ) return 1;
      else if ( _csc_ID == 9 && _neighbor == 1 ) return 2;
      else if ( _csc_ID == 3 && _neighbor == 0 ) return 3;
      else if ( _csc_ID == 9 && _neighbor == 0 ) return 4;
      else return 0;
    }
  }

  // Unpacks pT LUT address into dPhi, dTheta, CLCT, FR, eta, and mode   
  // Based on L1Trigger/L1TMuonEndCap/interface/PtAssignment.h
  // "Mode" here is the true mode, not the inverted mode used in PtAssignment.h
  void EMTFTrack::ImportPtLUT(int _mode, unsigned long _address) {
    if     (_mode == 12) { // mode_inv == 3
      EMTFTrack::set_dPhi_12     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_12 (dPhi_12 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_12   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_2      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_2 (clct_2   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_2        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 10) { // mode_inv == 5
      EMTFTrack::set_dPhi_13     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_13 (dPhi_13 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_13   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_3      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_3 (clct_3   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_3        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 9) { // mode_inv == 9
      EMTFTrack::set_dPhi_14     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_14 (dPhi_14 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_14   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_4      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_4 (clct_4   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_4        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 6) { // mode_inv = 6
      EMTFTrack::set_dPhi_23     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_23 (dPhi_23 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_23   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_2      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_2 (clct_2   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_3      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_3 (clct_3   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_2        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_3        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 5) { // mode_inv == 10
      EMTFTrack::set_dPhi_24     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_24 (dPhi_24 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_24   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_2      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_2 (clct_2   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_4      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_4 (clct_4   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_2        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_4        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 3) { // mode_inv == 12
      EMTFTrack::set_dPhi_34     ( ( _address >> (0) )   & ( (1 << 9) - 1) );
      EMTFTrack::set_dPhi_34 (dPhi_34 * (( ( _address >> (0+9) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_34   ( ( _address >> (0+9+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_3      ( ( _address >> (0+9+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_3 (clct_3   * ( ( ( _address >> (0+9+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_clct_4      ( ( _address >> (0+9+1+3+2+1) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_4 (clct_4   * ( ( ( _address >> (0+9+1+3+2+1+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_3        ( ( _address >> (0+9+1+3+2+1+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_fr_4        ( ( _address >> (0+9+1+3+2+1+2+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+9+1+3+2+1+2+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+9+1+3+2+1+2+1+1+1+5) ) & ( (1 << 4) - 1) );
    }      
    else if (_mode == 14) { // mode_inv == 7
      EMTFTrack::set_dPhi_12     ( ( _address >> (0) )     & ( (1 << 7) - 1) );
      EMTFTrack::set_dPhi_23     ( ( _address >> (0+7) )   & ( (1 << 5) - 1) );
      EMTFTrack::set_dPhi_12 (dPhi_12 * (( ( _address >> (0+7+5) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dPhi_23 (dPhi_23 * (( ( _address >> (0+7+5+1) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_13   ( ( _address >> (0+7+5+1+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+7+5+1+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+7+5+1+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+7+5+1+1+3+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+7+5+1+1+3+2+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+7+5+1+1+3+2+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 13) { // mode_inv == 11
      EMTFTrack::set_dPhi_12     ( ( _address >> (0) )     & ( (1 << 7) - 1) );
      EMTFTrack::set_dPhi_24     ( ( _address >> (0+7) )   & ( (1 << 5) - 1) );
      EMTFTrack::set_dPhi_12 (dPhi_12 * (( ( _address >> (0+7+5) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dPhi_24 (dPhi_24 * (( ( _address >> (0+7+5+1) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_14   ( ( _address >> (0+7+5+1+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+7+5+1+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+7+5+1+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+7+5+1+1+3+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+7+5+1+1+3+2+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+7+5+1+1+3+2+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 11) {
      EMTFTrack::set_dPhi_13     ( ( _address >> (0) )     & ( (1 << 7) - 1) );
      EMTFTrack::set_dPhi_34     ( ( _address >> (0+7) )   & ( (1 << 5) - 1) );
      EMTFTrack::set_dPhi_13 (dPhi_13 * (( ( _address >> (0+7+5) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dPhi_34 (dPhi_34 * (( ( _address >> (0+7+5+1) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_14   ( ( _address >> (0+7+5+1+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_1      ( ( _address >> (0+7+5+1+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_1 (clct_1   * ( ( ( _address >> (0+7+5+1+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+7+5+1+1+3+2+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+7+5+1+1+3+2+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+7+5+1+1+3+2+1+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 7) { // mode_inv == 14
      EMTFTrack::set_dPhi_23     ( ( _address >> (0) )     & ( (1 << 7) - 1) );
      EMTFTrack::set_dPhi_34     ( ( _address >> (0+7) )   & ( (1 << 6) - 1) );
      EMTFTrack::set_dPhi_23 (dPhi_23 * (( ( _address >> (0+7+6) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dPhi_34 (dPhi_34 * (( ( _address >> (0+7+6+1) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dTheta_24   ( ( _address >> (0+7+6+1+1) ) & ( (1 << 3) - 1) );
      EMTFTrack::set_clct_2      ( ( _address >> (0+7+5+1+1+3) ) & ( (1 << 2) - 1) );
      EMTFTrack::set_clct_2 (clct_2   * ( ( ( _address >> (0+7+6+1+1+3+2) ) & ( (1 << 1) - 1) ) == 0 ? -1 : 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+7+6+1+1+3+2+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+7+6+1+1+3+2+1+5) ) & ( (1 << 4) - 1) );
    }
    else if (_mode == 15) { // mode_inv == 15
      EMTFTrack::set_dPhi_12     ( ( _address >> (0) )     & ( (1 << 7) - 1) );
      EMTFTrack::set_dPhi_23     ( ( _address >> (0+7) )   & ( (1 << 5) - 1) );
      EMTFTrack::set_dPhi_34     ( ( _address >> (0+7+5) ) & ( (1 << 6) - 1) );
      EMTFTrack::set_dPhi_23 (dPhi_23 * (( ( _address >> (0+7+5+6) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_dPhi_34 (dPhi_34 * (( ( _address >> (0+7+5+6+1) ) & ( (1 << 1) - 1 ) ) == 0 ? -1 : 1) );
      EMTFTrack::set_fr_1        ( ( _address >> (0+7+5+6+1+1) ) & ( (1 << 1) - 1) );
      EMTFTrack::set_eta_LUT     ( ( _address >> (0+7+5+6+1+1+1) ) & ( (1 << 5) - 1) );
      EMTFTrack::set_mode_LUT    ( ( _address >> (0+7+5+6+1+1+1+5) ) & ( (1 << 4) - 1) );
    }

  } // End function: void EMTFTrack::ImportPtLUT

  EMTFTrack EMTFTrackExtra::CreateEMTFTrack() {

    EMTFTrack thisTrack;
    for (int iHit = 0; iHit < NumHitsExtra(); iHit++) {
      thisTrack.push_Hit( _HitsExtra.at(iHit).CreateEMTFHit() );
    }
      
    thisTrack.set_endcap        ( Endcap()       );
    thisTrack.set_sector        ( Sector()       );
    thisTrack.set_sector_GMT    ( Sector_GMT()   );
    thisTrack.set_sector_index  ( Sector_index() );
    thisTrack.set_mode          ( Mode()         );
    thisTrack.set_mode_LUT      ( Mode_LUT()     );
    thisTrack.set_quality       ( Quality()      );
    thisTrack.set_bx            ( BX()           );
    thisTrack.set_pt            ( Pt()           );
    thisTrack.set_pt_GMT        ( Pt_GMT()       );
    thisTrack.set_pt_LUT_addr   ( Pt_LUT_addr()  );
    thisTrack.set_eta           ( Eta()          );
    thisTrack.set_eta_GMT       ( Eta_GMT()      );
    thisTrack.set_eta_LUT       ( Eta_LUT()      );
    thisTrack.set_phi_loc_int   ( Phi_loc_int()  );
    thisTrack.set_phi_loc_deg   ( Phi_loc_deg()  );
    thisTrack.set_phi_loc_rad   ( Phi_loc_rad()  );
    thisTrack.set_phi_GMT       ( Phi_GMT()      );
    thisTrack.set_phi_glob_deg  ( Phi_glob_deg() );
    thisTrack.set_phi_glob_rad  ( Phi_glob_rad() );
    thisTrack.set_charge        ( Charge()       );
    thisTrack.set_charge_GMT    ( Charge_GMT()   );
    thisTrack.set_charge_valid  ( Charge_valid() );
    thisTrack.set_dPhi_12       ( DPhi_12()      );
    thisTrack.set_dPhi_13       ( DPhi_13()      );
    thisTrack.set_dPhi_14       ( DPhi_14()      );
    thisTrack.set_dPhi_23       ( DPhi_23()      );
    thisTrack.set_dPhi_24       ( DPhi_24()      );
    thisTrack.set_dPhi_34       ( DPhi_34()      );
    thisTrack.set_dTheta_12     ( DTheta_12()    );
    thisTrack.set_dTheta_13     ( DTheta_13()    );
    thisTrack.set_dTheta_14     ( DTheta_14()    );
    thisTrack.set_dTheta_23     ( DTheta_23()    );
    thisTrack.set_dTheta_24     ( DTheta_24()    );
    thisTrack.set_dTheta_34     ( DTheta_34()    );
    thisTrack.set_clct_1        ( CLCT_1()       );
    thisTrack.set_clct_2        ( CLCT_2()       );
    thisTrack.set_clct_3        ( CLCT_3()       );
    thisTrack.set_clct_4        ( CLCT_4()       );
    thisTrack.set_fr_1          ( FR_1()         );
    thisTrack.set_fr_2          ( FR_2()         );
    thisTrack.set_fr_3          ( FR_3()         );
    thisTrack.set_fr_4          ( FR_4()         );
    thisTrack.set_track_num     ( Track_num()    );
    thisTrack.set_has_neighbor  ( Has_neighbor() );
    thisTrack.set_all_neighbor  ( All_neighbor() );

    return thisTrack;

  } // End EMTFTrackExtra::CreateEMTFTrack

    
} // End namespace l1t
