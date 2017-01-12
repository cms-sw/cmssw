
#include "L1Trigger/L1TMuonEndCap/interface/EMTFHitTools.h"

namespace l1t {

  void EMTFHit::PrintSimulatorHeader() {
    std::cout << "Simulator hits: time_bin, endcap, sector, subsector, station, valid, "
              << "quality, CLCT pattern, wiregroup,  cscid, bend,  halfstrip" << std::endl;
    std::cout << "Expected values:   0 - 7, 1 or 0,  1 - 6, 0 / 1 - 2,   1 - 4, 0 - 1, "
              << " 0 - 15,       0 - 15,     0 - ?, 1 - 18, 0 - 1,     0 - ?" << std::endl;
  }

  void EMTFHit::PrintForSimulator () {
    std::cout << bx + 6 << ", " << ((endcap == 1) ? 1 : 0) << ", " << sector << ", " << std::max(subsector, 0) << ", "
	      << station << ", " << valid << ", " << quality << ", " << pattern << ", " << wire << ", "
              << ((ring == 4) ? csc_ID + 9 : csc_ID) << ", " << std::max(bend, 0) << ", " << strip << std::endl;
  }

  // Based on L1Trigger/L1TMuon/src/MuonTriggerPrimitive.cc
  // TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi)
  void EMTFHit::ImportCSCDetId( const CSCDetId& _detId) {

    EMTFHit::SetCSCDetId ( _detId ); 
    // It appears the following function *actually does literally nothing* - AWB 17.03.16
    // calculateCSCGlobalSector(detid,_globalsector,_subsector);

    // Based on L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h
    EMTFHit::set_endcap  ( (_detId.endcap() == 2) ? -1 : _detId.endcap() ); // Convert from {+,-} = {1,2} to {1,-1}
    EMTFHit::set_station ( _detId.station()       );
    EMTFHit::set_sector  ( _detId.triggerSector() );
    EMTFHit::set_ring    ( _detId.ring()          );
    EMTFHit::set_chamber ( _detId.chamber()       );

    EMTFHit::set_is_CSC_hit ( 1 );
    EMTFHit::set_is_RPC_hit ( 0 );

  } // End EMTFHit::ImportCSCDetId

  CSCDetId EMTFHit::CreateCSCDetId() {

    return CSCDetId( (endcap == 1) ? 1 : 2, station,    // For now, leave "layer" unfilled, defaults to 0.
  		     (ring == 4) ? 1 : ring, chamber ); // Not sure if this is correct, or what "layer" does. - AWB 27.04.16
  }

  // Based on L1Trigger/L1TMuon/src/MuonTriggerPrimitive.cc
  // TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi)
  // This is what gets filled when "getCSCData()" is called in
  // L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h
  void EMTFHit::ImportCSCCorrelatedLCTDigi( const CSCCorrelatedLCTDigi& _digi ) {

    EMTFHit::SetCSCLCTDigi ( _digi );

    EMTFHit::set_track_num ( _digi.getTrknmb()  );
    EMTFHit::set_valid     ( _digi.isValid()    );
    EMTFHit::set_quality   ( _digi.getQuality() );
    EMTFHit::set_wire      ( _digi.getKeyWG()   );
    EMTFHit::set_strip     ( _digi.getStrip()   );
    EMTFHit::set_pattern   ( _digi.getPattern() );
    EMTFHit::set_bend      ( _digi.getBend()    );
    EMTFHit::set_bx        ( _digi.getBX() - 6  ); // Standard for csctfDigis in data, simCscTriggerPrimitiveDigis in MC
    EMTFHit::set_mpc_link  ( _digi.getMPCLink() );
    EMTFHit::set_sync_err  ( _digi.getSyncErr() );
    EMTFHit::set_csc_ID    ( _digi.getCSCID()   );

    EMTFHit::set_subsector ( calc_subsector( station, chamber ) ); 

  } // End EMTFHit::ImportCSCCorrelatedLCTDigi

  void EMTFHitExtra::ImportCSCCorrelatedLCTDigi( const CSCCorrelatedLCTDigi& _digi ) { 

    EMTFHit::ImportCSCCorrelatedLCTDigi ( _digi );
    EMTFHitExtra::set_bx0  ( _digi.getBX0()     );

  } // End EMTFHitExtra::ImportCSCCorrelatedLCTDigi

  CSCCorrelatedLCTDigi EMTFHit::CreateCSCCorrelatedLCTDigi() {

    return CSCCorrelatedLCTDigi( 1, valid, quality, wire, strip, 
  				 pattern, (bend == 1) ? 1 : 0,   
  				 bx + 6, 0, 0, sync_err, csc_ID );  
    // Unsure of how to fill "trknmb" or "bx0" - for now filling with 1 and 0. - AWB 27.04.16
    // Appear to be unused in the emulator code. mpclink = 0 (after bx) indicates unsorted.
  }

  void EMTFHit::ImportME( const emtf::ME _ME) {

    EMTFHit::set_wire       ( _ME.Wire() );
    EMTFHit::set_strip      ( _ME.Strip() );
    EMTFHit::set_quality    ( _ME.Quality() );
    EMTFHit::set_pattern    ( _ME.CLCT_pattern() );
    EMTFHit::set_bend       ( (_ME.LR() == 1) ? 1 : -1 );
    EMTFHit::set_valid      ( _ME.VP() );
    EMTFHit::set_sync_err   ( _ME.SE() );
    EMTFHit::set_bx         ( _ME.TBIN() - 3 );
    EMTFHit::set_bc0        ( _ME.BC0() ); 
    EMTFHit::set_is_CSC_hit ( true  );
    EMTFHit::set_is_RPC_hit ( false );

    // Station, CSC_ID, Sector, Subsector, Neighbor, Sector_index, Ring, and Chamber filled in
    // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockME.cc
    // "set_layer()" is not invoked, so Layer is not yet filled - AWB 21.04.16

  } // End EMTFHit::ImportME

  int calc_ring (int _station, int _csc_ID, int _strip) {
    if (_station > 1) {
      if      (_csc_ID <  4) return 1;
      else if (_csc_ID < 10) return 2;
      else return -999;
    }
    else if (_station == 1) {
      if      (_csc_ID < 4 && _strip > 127) return 4;
      else if (_csc_ID < 4 && _strip >=  0) return 1;
      else if (_csc_ID > 3 && _csc_ID <  7) return 2;
      else if (_csc_ID > 6 && _csc_ID < 10) return 3;
      else return -999;
    }
    else return -999;
  } // End EMTFHit::calc_ring

  int calc_chamber (int _station, int _sector, int _subsector, int _ring, int _csc_ID) {
    int tmp_chamber = -999;
    if (_station == 1) {
      tmp_chamber = ((_sector-1) * 6) + _csc_ID + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
      if (_ring == 2)       tmp_chamber -= 3;
      if (_ring == 3)       tmp_chamber -= 6;
      if (_subsector == 2)  tmp_chamber += 3;
      if (tmp_chamber > 36) tmp_chamber -= 36;
    }
    else if (_ring == 1) { 
      tmp_chamber = ((_sector-1) * 3) + _csc_ID + 1; // Chamber offset of 1: First chamber in sector 1 is chamber 2
      if (tmp_chamber > 18) tmp_chamber -= 18;
    }
    else if (_ring == 2) {
      tmp_chamber = ((_sector-1) * 6) + _csc_ID - 3 + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
      if (tmp_chamber > 36) tmp_chamber -= 36;
    }
    return tmp_chamber;
  } // End EMTFHit::calc_chamber

  EMTFHit EMTFHitExtra::CreateEMTFHit() {

    EMTFHit thisHit;
    thisHit.set_endcap       ( Endcap()        );
    thisHit.set_station      ( Station()       );
    thisHit.set_ring         ( Ring()          );
    thisHit.set_sector       ( Sector()        );
    thisHit.set_sector_index ( Sector_index()  );
    thisHit.set_subsector    ( Subsector()     );
    thisHit.set_chamber      ( Chamber()       );
    thisHit.set_csc_ID       ( CSC_ID()        );
    thisHit.set_neighbor     ( Neighbor()      );
    thisHit.set_mpc_link     ( MPC_link()      );
    thisHit.set_wire         ( Wire()          );
    thisHit.set_strip        ( Strip()         );
    thisHit.set_track_num    ( Track_num()     );
    thisHit.set_quality      ( Quality()       );
    thisHit.set_pattern      ( Pattern()       );
    thisHit.set_bend         ( Bend()          );
    thisHit.set_valid        ( Valid()         );
    thisHit.set_sync_err     ( Sync_err()      );
    thisHit.set_bc0          ( BC0()           );
    thisHit.set_bx           ( BX()            );
    thisHit.set_stub_num     ( Stub_num()      );
    thisHit.set_is_CSC_hit   ( Is_CSC_hit()    );
    thisHit.set_is_RPC_hit   ( Is_RPC_hit()    );

    return thisHit;
  } // End EMTFHitExtra::CreateEMTFHit

    
} // End namespace l1t
