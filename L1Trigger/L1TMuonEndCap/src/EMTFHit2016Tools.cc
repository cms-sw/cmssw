
#include "L1Trigger/L1TMuonEndCap/interface/EMTFHit2016Tools.h"

namespace l1t {

  void EMTFHit2016::PrintSimulatorHeader() {
    std::cout << "Simulator hits: time_bin, endcap, sector, subsector, station, valid, "
              << "quality, CLCT pattern, wiregroup,  cscid, bend,  halfstrip" << std::endl;
    std::cout << "Expected values:   0 - 7, 1 or 0,  1 - 6, 0 / 1 - 2,   1 - 4, 0 - 1, "
              << " 0 - 15,       0 - 15,     0 - ?, 1 - 18, 0 - 1,     0 - ?" << std::endl;
  }

  void EMTFHit2016::PrintForSimulator () {
    std::cout << bx + 6 << ", " << ((endcap == 1) ? 1 : 0) << ", " << sector << ", " << std::max(subsector, 0) << ", "
	      << station << ", " << valid << ", " << quality << ", " << pattern << ", " << wire << ", "
              << ((ring == 4) ? csc_ID + 9 : csc_ID) << ", " << std::max(bend, 0) << ", " << strip << std::endl;
  }

  // Based on L1Trigger/L1TMuon/src/MuonTriggerPrimitive.cc
  // TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi)
  void EMTFHit2016::ImportCSCDetId( const CSCDetId& _detId) {

    EMTFHit2016::SetCSCDetId ( _detId ); 
    // It appears the following function *actually does literally nothing* - AWB 17.03.16
    // calculateCSCGlobalSector(detid,_globalsector,_subsector);

    // Based on L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h
    EMTFHit2016::set_endcap  ( (_detId.endcap() == 2) ? -1 : _detId.endcap() ); // Convert from {+,-} = {1,2} to {1,-1}
    EMTFHit2016::set_station ( _detId.station()       );
    EMTFHit2016::set_sector  ( _detId.triggerSector() );
    EMTFHit2016::set_ring    ( _detId.ring()          );
    EMTFHit2016::set_chamber ( _detId.chamber()       );

    EMTFHit2016::set_is_CSC_hit ( 1 );
    EMTFHit2016::set_is_RPC_hit ( 0 );

  } // End EMTFHit2016::ImportCSCDetId

  CSCDetId EMTFHit2016::CreateCSCDetId() {

    return CSCDetId( (endcap == 1) ? 1 : 2, station,    // For now, leave "layer" unfilled, defaults to 0.
  		     (ring == 4) ? 1 : ring, chamber ); // Not sure if this is correct, or what "layer" does. - AWB 27.04.16
  }

  void EMTFHit2016::ImportRPCDetId( const RPCDetId& _detId) {

    EMTFHit2016::SetRPCDetId ( _detId ); 
    
    EMTFHit2016::set_endcap    ( _detId.region()    ); // 0 for barrel, +/-1 for +/- endcap
    EMTFHit2016::set_station   ( _detId.station()   ); // Same as in CSCs (?)
    EMTFHit2016::set_sector    ( _detId.sector()    ); // Same as in CSCs (?)  
    EMTFHit2016::set_subsector ( _detId.subsector() ); // Same as in CSCs (?)
    EMTFHit2016::set_ring      ( _detId.ring()      ); // Ring number in endcap (from 1 to 3, but only 2 and 3 exist currently)
    EMTFHit2016::set_roll      ( _detId.roll()      ); // AKA eta "partition" or "segment": subdivision of ring into 3 parts, noted "C-B-A" in-to-out

    EMTFHit2016::set_is_CSC_hit ( 0 );
    EMTFHit2016::set_is_RPC_hit ( 1 );

  } // End EMTFHit2016::ImportCSCDetId

  RPCDetId EMTFHit2016::CreateRPCDetId() {
    
    return RPCDetId( endcap, ring, station, sector, rpc_layer, subsector, roll );
    
  }

  // Based on L1Trigger/L1TMuon/src/MuonTriggerPrimitive.cc
  // TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi)
  // This is what gets filled when "getCSCData()" is called in
  // L1Trigger/L1TMuonEndCap/interface/PrimitiveConverter.h
  void EMTFHit2016::ImportCSCCorrelatedLCTDigi( const CSCCorrelatedLCTDigi& _digi ) {

    EMTFHit2016::SetCSCLCTDigi ( _digi );

    EMTFHit2016::set_track_num ( _digi.getTrknmb()  );
    EMTFHit2016::set_valid     ( _digi.isValid()    );
    EMTFHit2016::set_quality   ( _digi.getQuality() );
    EMTFHit2016::set_wire      ( _digi.getKeyWG()   );
    EMTFHit2016::set_strip     ( _digi.getStrip()   );
    EMTFHit2016::set_pattern   ( _digi.getPattern() );
    EMTFHit2016::set_bend      ( _digi.getBend()    );
    EMTFHit2016::set_bx        ( _digi.getBX() - 6  ); // Standard for csctfDigis in data, simCscTriggerPrimitiveDigis in MC
    EMTFHit2016::set_mpc_link  ( _digi.getMPCLink() );
    EMTFHit2016::set_sync_err  ( _digi.getSyncErr() );
    EMTFHit2016::set_csc_ID    ( _digi.getCSCID()   );

    EMTFHit2016::set_subsector ( calc_subsector( station, chamber ) ); 

  } // End EMTFHit2016::ImportCSCCorrelatedLCTDigi

  void EMTFHit2016Extra::ImportCSCCorrelatedLCTDigi( const CSCCorrelatedLCTDigi& _digi ) { 

    EMTFHit2016::ImportCSCCorrelatedLCTDigi ( _digi );
    EMTFHit2016Extra::set_bx0  ( _digi.getBX0()     );

  } // End EMTFHit2016Extra::ImportCSCCorrelatedLCTDigi

  CSCCorrelatedLCTDigi EMTFHit2016::CreateCSCCorrelatedLCTDigi() {

    return CSCCorrelatedLCTDigi( 1, valid, quality, wire, strip, 
  				 pattern, (bend == 1) ? 1 : 0,   
  				 bx + 6, 0, 0, sync_err, csc_ID );  
    // Unsure of how to fill "trknmb" or "bx0" - for now filling with 1 and 0. - AWB 27.04.16
    // Appear to be unused in the emulator code. mpclink = 0 (after bx) indicates unsorted.
  }

  void EMTFHit2016::ImportRPCDigi( const RPCDigi& _digi ) {

    EMTFHit2016::SetRPCDigi    ( _digi );
    EMTFHit2016::set_strip_hi  ( _digi.strip()  );
    EMTFHit2016::set_strip_low ( _digi.strip()  );
    EMTFHit2016::set_bx        ( _digi.bx() - 6 );  // Started looking at RPCs, not used yet

  }

  RPCDigi EMTFHit2016::CreateRPCDigi() {
    return RPCDigi( strip, bx + 6 );
  }

  void EMTFHit2016::ImportME( const emtf::ME _ME) {

    EMTFHit2016::set_wire       ( _ME.Wire() );
    EMTFHit2016::set_strip      ( _ME.Strip() );
    EMTFHit2016::set_quality    ( _ME.Quality() );
    EMTFHit2016::set_pattern    ( _ME.CLCT_pattern() );
    EMTFHit2016::set_bend       ( (_ME.LR() == 1) ? 1 : -1 );
    EMTFHit2016::set_valid      ( _ME.VP() );
    EMTFHit2016::set_sync_err   ( _ME.SE() );
    EMTFHit2016::set_bx         ( _ME.TBIN() - 3 );
    EMTFHit2016::set_bc0        ( _ME.BC0() ); 
    EMTFHit2016::set_is_CSC_hit ( true  );
    EMTFHit2016::set_is_RPC_hit ( false );

    // Station, CSC_ID, Sector, Subsector, Neighbor, Sector_index, Ring, and Chamber filled in
    // EventFilter/L1TRawToDigi/src/implementations_stage2/EMTFBlockME.cc
    // "set_layer()" is not invoked, so Layer is not yet filled - AWB 21.04.16

  } // End EMTFHit2016::ImportME

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
  } // End EMTFHit2016::calc_ring

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
  } // End EMTFHit2016::calc_chamber

  EMTFHit2016 EMTFHit2016Extra::CreateEMTFHit2016() {

    EMTFHit2016 thisHit;
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
  } // End EMTFHit2016Extra::CreateEMTFHit2016

    
} // End namespace l1t
