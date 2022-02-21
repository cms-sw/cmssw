#ifndef RecoTracker_MkFitCore_interface_cms_common_macros_h
#define RecoTracker_MkFitCore_interface_cms_common_macros_h

#ifdef MKFIT_STANDALONE
#define CMS_SA_ALLOW
#else
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#endif

#endif
