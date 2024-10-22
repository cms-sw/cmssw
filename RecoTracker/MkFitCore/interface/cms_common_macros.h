#ifndef RecoTracker_MkFitCore_interface_cms_common_macros_h
#define RecoTracker_MkFitCore_interface_cms_common_macros_h

#ifdef MKFIT_STANDALONE
#define CMS_SA_ALLOW
#else
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/isFinite.h"
#endif

namespace mkfit {

  constexpr bool isFinite(float x) {
#ifdef MKFIT_STANDALONE
    const unsigned int mask = 0x7f800000;
    union {
      unsigned int l;
      float d;
    } v = {.d = x};
    return (v.l & mask) != mask;
#else
    return edm::isFinite(x);
#endif
  }

}  // namespace mkfit

#endif
