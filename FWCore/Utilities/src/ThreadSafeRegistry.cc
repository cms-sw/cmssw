// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     ThreadSafeRegistry
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue, 22 Jul 2014 21:06:25 GMT
//

// system include files
#include <mutex>

// user include files
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"

namespace edm {
  namespace detail {
    [[edm::thread_safe]] static std::mutex registry_mutex;
  }
}
