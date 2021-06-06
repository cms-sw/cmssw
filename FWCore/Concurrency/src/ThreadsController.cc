// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     ThreadsController
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 28 Aug 2020 19:42:30 GMT
//

// system include files

// user include files
#include "FWCore/Concurrency/interface/ThreadsController.h"

namespace edm {
  std::unique_ptr<tbb::global_control> ThreadsController::makeStackSize(size_t iStackSize) {
    return std::make_unique<tbb::global_control>(tbb::global_control::thread_stack_size, iStackSize);
  }

}  // namespace edm
