#ifndef L1Trigger_L1TMuonEndCapPhase2_DebugUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_DebugUtils_h

#include <cassert>
#include <string>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

// Uncomment the following line to use assert
#define EMTF_ALLOW_ASSERT

#ifdef EMTF_ALLOW_ASSERT
#define emtf_assert(expr) (assert(expr))
#else
#define emtf_assert(expr) ((void)(expr))
#endif

#endif // namespace L1Trigger_L1TMuonEndCapPhase2_DebugUtils_h

