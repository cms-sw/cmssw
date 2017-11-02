// Trigger Primitive Converter for RPC hits
//
// Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
// and outputs vector of 'ConvertedHits'

#ifndef L1Trigger_L1TMuonEndCap_PrimitiveConverterRPC_h
#define L1Trigger_L1TMuonEndCap_PrimitiveConverterRPC_h

//

#include <iostream>

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFHit2016Tools.h"

class PrimitiveConverterRPC {
 public:
  PrimitiveConverterRPC();
  l1t::EMTFHit2016ExtraCollection convert(std::vector<L1TMuon::TriggerPrimitive> TriggPrim, int SectIndex, edm::ESHandle<RPCGeometry> rpc_geom);
  std::vector<ConvertedHit> fillConvHits(l1t::EMTFHit2016ExtraCollection exHits);
  bool sameRpcChamber(l1t::EMTFHit2016Extra hitA, l1t::EMTFHit2016Extra hitB);

 private:

};

#endif /* #define L1Trigger_L1TMuonEndCap_PrimitiveConverterRPC_h */
