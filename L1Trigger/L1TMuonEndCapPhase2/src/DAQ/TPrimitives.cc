#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"

using namespace emtf::phase2;

TPEntry::TPEntry(const TPEntry& tp_entry) : tp_(tp_entry.tp_), info_(tp_entry.info_) {
  // Do nothing
}

TPEntry::TPEntry(const TriggerPrimitive& tp) : tp_(tp), info_() {
  // Do nothing
}

TPEntry::TPEntry(const TriggerPrimitive& tp, const TPInfo& tp_info) : tp_(tp), info_(tp_info) {
  // Do nothing
}

TPEntry::TPEntry(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi) : tp_(detid, digi), info_() {
  // Do nothing
}

TPEntry::TPEntry(const RPCDetId& detid, const RPCRecHit& rechit) : tp_(detid, rechit), info_() {
  // Do nothing
}

TPEntry::TPEntry(const GEMDetId& detid, const GEMPadDigiCluster& digi) : tp_(detid, digi), info_() {
  // Do nothing
}

TPEntry::TPEntry(const ME0DetId& detid, const ME0TriggerDigi& digi) : tp_(detid, digi), info_() {
  // Do nothing
}

TPEntry::TPEntry(const GEMDetId& detid, const ME0TriggerDigi& digi) : tp_(detid, digi), info_() {
  // Do nothing
}

TPEntry::~TPEntry() {
  // Do nothing
}
