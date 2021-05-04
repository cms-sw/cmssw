#ifndef L1TMuonEndCap_DebugTools_h
#define L1TMuonEndCap_DebugTools_h

#include <cassert>

#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFRoad.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitiveFwd.h"

// Uncomment the following line to use assert
//#define EMTF_ALLOW_ASSERT

#ifdef EMTF_ALLOW_ASSERT
#define emtf_assert(expr) (assert(expr))
#else
#define emtf_assert(expr) ((void)(expr))
#endif

namespace emtf {

  void dump_fw_raw_input(const l1t::EMTFHitCollection& out_hits, const l1t::EMTFTrackCollection& out_tracks);

}  // namespace emtf

#endif
