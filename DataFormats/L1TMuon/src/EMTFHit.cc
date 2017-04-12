#include "DataFormats/L1TMuon/interface/EMTFHit.h"

namespace l1t {

  CSCDetId EMTFHit::CreateCSCDetId() const {
    return CSCDetId( (endcap == 1) ? 1 : 2, station,    // For now, leave "layer" unfilled, defaults to 0.
         (ring == 4) ? 1 : ring, chamber ); // Not sure if this is correct, or what "layer" does. - AWB 27.04.16
  }

  // // Not yet implemented - AWB 15.03.17
  // RPCDetId EMTFHit::CreateRPCDetId() const {
  //   return RPCDetId( endcap, ring, station, sector, rpc_layer, subsector, roll );
  // }

  CSCCorrelatedLCTDigi EMTFHit::CreateCSCCorrelatedLCTDigi() const {
    return CSCCorrelatedLCTDigi( 1, valid, quality, wire, strip,
         pattern, (bend == 1) ? 1 : 0,
         bx + 6, 0, 0, sync_err, csc_ID );
    // Unsure of how to fill "trknmb" or "bx0" - for now filling with 1 and 0. - AWB 27.04.16
    // Appear to be unused in the emulator code. mpclink = 0 (after bx) indicates unsorted.
  }

  // // Not yet implemented - AWB 15.03.17
  // RPCDigi EMTFHit::CreateRPCDigi() const {
  //   return RPCDigi( (strip_hi + strip_lo) / 2, bx + 6 );
  // }

} // End namespace l1t
