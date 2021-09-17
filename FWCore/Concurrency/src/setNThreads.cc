//
//  setNThreads.cc
//  CMSSW
//
//  Created by Chris Jones on 7/24/20.
//
#include "tbb/task_arena.h"
#include "FWCore/Concurrency/interface/setNThreads.h"

namespace edm {
  unsigned int setNThreads(unsigned int iNThreads, unsigned int iStackSize, std::unique_ptr<ThreadsController>& oPtr) {
    //stack size is given in KB but passed in as bytes
    iStackSize *= 1024;

    oPtr.reset();
    if (0 == iNThreads) {
      //Allow TBB to decide how many threads. This is normally the number of CPUs in the machine.
      iNThreads = tbb::this_task_arena::max_concurrency();
    }
    oPtr = std::make_unique<ThreadsController>(static_cast<int>(iNThreads), iStackSize);

    return iNThreads;
  }
}  // namespace edm
