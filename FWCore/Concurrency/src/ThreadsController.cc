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
#include "tbb/task_scheduler_init.h"

// user include files
#include "FWCore/Concurrency/interface/ThreadsController.h"

//NOTE: The only way at present to oversubscribe the number of TBB threads to cores is to
// use a tbb::task_scheduler_init.
namespace edm {

  void ThreadsController::Destructor::operator()(void* iThis) const {
    delete static_cast<tbb::task_scheduler_init*>(iThis);
  }

  std::unique_ptr<void, ThreadsController::Destructor> ThreadsController::makeOversubscriber(size_t iNThreads) {
    return std::unique_ptr<void, ThreadsController::Destructor>(
        static_cast<void*>(new tbb::task_scheduler_init(iNThreads)));
  }

}  // namespace edm
