#ifndef L1TMuonEndCap_EMTFGEMDetIdImpl_h
#define L1TMuonEndCap_EMTFGEMDetIdImpl_h

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFGEMDetId.h"

namespace emtf {

  template<typename T=void>
  EMTFGEMDetId construct_EMTFGEMDetId(const L1TMuon::TriggerPrimitive& tp) {
    if (!tp.getGEMData().isME0) {
      GEMDetId id(tp.detId<GEMDetId>());
      return EMTFGEMDetId(id);
    } else {
      ME0DetId id(tp.detId<ME0DetId>());
      return EMTFGEMDetId(id);
    }
  };

}  // namespace emtf

#endif
