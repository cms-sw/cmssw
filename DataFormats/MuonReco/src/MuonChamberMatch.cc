#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
using namespace reco;

int MuonChamberMatch::station()  const {
   if( detector() ==  MuonSubdetId::DT ) {    // DT
      DTChamberId segId(id.rawId());
      return segId.station();
   }
   if( detector() == MuonSubdetId::CSC ) {    // CSC
      CSCDetId segId(id.rawId());
      return segId.station();
   }
   if( detector() == MuonSubdetId::RPC ) {    //RPC
      RPCDetId segId(id.rawId());
      return segId.station();
   }
   return -1; // is this appropriate? fix this
}
