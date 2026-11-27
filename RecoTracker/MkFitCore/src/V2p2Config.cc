// Standalone definitions of the knobs declared in V2p2Config.h. In the CMSSW
// build they are constexpr and this file defines nothing.
//
// Must be the first include, so the header is expanded with definitions.
#define MKFIT_V2P2_CONFIG_DEFINE
#include "RecoTracker/MkFitCore/src/V2p2Config.h"

#if defined(MKFIT_STANDALONE)

namespace mkfit::Config::V2p2 {

  void set_extra_dq(float f) {
    Window::dq_trk_fac = f;
    Window::dq_hit_fac = f * 1.2f;  // the old EXTRA_DQ ratio
  }

}  // namespace mkfit::Config::V2p2

#endif
