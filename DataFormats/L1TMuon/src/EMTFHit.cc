#include "DataFormats/L1TMuon/interface/EMTFHit.h"

namespace l1t {

  CSCDetId EMTFHit::CreateCSCDetId() const {
    return CSCDetId( (endcap == 1) ? 1 : 2, station,
		     (ring == 4) ? 1 : ring, chamber, 0 ); 
    // Layer always filled as 0 (indicates "whole chamber")
    // See http://cmslxr.fnal.gov/source/L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.cc#0198
  }
  
  // // Not yet implemented - AWB 15.03.17
  // RPCDetId EMTFHit::CreateRPCDetId() const {
  //   return RPCDetId( endcap, ring, station, sector, rpc_layer, subsector, roll );
  // }

  CSCCorrelatedLCTDigi EMTFHit::CreateCSCCorrelatedLCTDigi() const {
    return CSCCorrelatedLCTDigi( 1, valid, quality, wire, strip,
				 pattern, (bend == 1) ? 1 : 0,
				 bx + 6, 0, 0, sync_err, csc_ID );
    // Filling "trknmb" with 1 and "bx0" with 0 (as in MC).
    // May consider filling "trknmb" with 2 for 2nd LCT in the same chamber. - AWB 24.05.17
    // trknmb and bx0 are unused in the EMTF emulator code. mpclink = 0 (after bx) indicates unsorted.
  }
  
  // // Not yet implemented - AWB 15.03.17
  // RPCDigi EMTFHit::CreateRPCDigi() const {
  //   return RPCDigi( (strip_hi + strip_lo) / 2, bx + 6 );
  // }
  
} // End namespace l1t
