// Trigger Primitive Converter for RPC hits
//
// Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
// and outputs vector of 'ConvertedHits'

#ifndef ADD_PrimitiveConverterRPC
#define ADD_PrimitiveConverterRPC

//

#include <iostream>

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFHitTools.h"

class PrimitiveConverterRPC {
 public:
  PrimitiveConverterRPC();
  l1t::EMTFHitExtraCollection convert(std::vector<L1TMuon::TriggerPrimitive> TriggPrim, int SectIndex, edm::ESHandle<RPCGeometry> rpc_geom);
  std::vector<ConvertedHit> fillConvHits(l1t::EMTFHitExtraCollection exHits);
  bool sameRpcChamber(l1t::EMTFHitExtra hitA, l1t::EMTFHitExtra hitB);

 private:

};

#endif
