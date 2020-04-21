#ifndef L1TMuonEndCap_DebugTools_h
#define L1TMuonEndCap_DebugTools_h

#include <cassert>

// Uncomment the following line to use assert
#define EMTF_ALLOW_ASSERT

#ifdef EMTF_ALLOW_ASSERT
#define emtf_assert(expr) (assert(expr))
#else
#define emtf_assert(expr) ((void)(expr))
#endif

namespace emtf {}  // namespace emtf

#endif
