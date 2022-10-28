#ifndef FWCore_Concurrency_ThreadsController_h
#define FWCore_Concurrency_ThreadsController_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     ThreadsController
//
/**\class ThreadsController ThreadsController.h "ThreadsController.h"

 Description: Controls how many threads and how much stack memory per thread

 Usage:
    The lifetime of the ThreadsController sets how long the options are in use.

*/
//
// Original Author:  FWCore
//         Created:  Fri, 18 Nov 2016 20:30:42 GMT
//

// system include files
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/task_arena.h"
#include <memory>

// user include files

// forward declarations

namespace edm {
  class ThreadsController {
  public:
    ThreadsController() = delete;
    explicit ThreadsController(unsigned int iNThreads)
        : m_nThreads{oneapi::tbb::global_control::max_allowed_parallelism, iNThreads}, m_stackSize{} {}
    ThreadsController(unsigned int iNThreads, size_t iStackSize)
        : m_nThreads{oneapi::tbb::global_control::max_allowed_parallelism, iNThreads},
          m_stackSize{makeStackSize(iStackSize)} {}

    // ---------- member functions ---------------------------
    void setStackSize(size_t iStackSize) { m_stackSize = makeStackSize(iStackSize); }

  private:
    static std::unique_ptr<oneapi::tbb::global_control> makeStackSize(size_t iStackSize);

    // ---------- member data --------------------------------
    oneapi::tbb::global_control m_nThreads;
    std::unique_ptr<oneapi::tbb::global_control> m_stackSize;
  };
}  // namespace edm

#endif
