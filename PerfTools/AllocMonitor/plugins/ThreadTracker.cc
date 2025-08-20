// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     ThreadTracker
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 11 Nov 2024 22:54:23 GMT
//

// system include files
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// user include files
#include "ThreadTracker.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

namespace cms::perftools::allocMon {
  ThreadTracker& ThreadTracker::instance() {
    CMS_THREAD_SAFE static ThreadTracker s_tracker;
    return s_tracker;
  }
}  // namespace cms::perftools::allocMon
