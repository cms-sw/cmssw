//
//  setNThreads.cc
//  CMSSW
//
//  Created by Chris Jones on 7/24/20.
//

#include "FWCore/Concurrency/interface/setNThreads.h"

namespace edm {
  unsigned int setNThreads(unsigned int iNThreads,
                           unsigned int iStackSize,
                           std::unique_ptr<tbb::task_scheduler_init>& oPtr) {
    //The TBB documentation doesn't explicitly say this, but when the task_scheduler_init's
    // destructor is run it does a 'wait all' for all tasks to finish and then shuts down all the threads.
    // This provides a clean synchronization point.
    //We have to destroy the old scheduler before starting a new one in order to
    // get tbb to actually switch the number of threads. If we do not, tbb stays at 1 threads

    //stack size is given in KB but passed in as bytes
    iStackSize *= 1024;

    oPtr.reset();
    if (0 == iNThreads) {
      //Allow TBB to decide how many threads. This is normally the number of CPUs in the machine.
      iNThreads = tbb::task_scheduler_init::default_num_threads();
    }
    oPtr = std::make_unique<tbb::task_scheduler_init>(static_cast<int>(iNThreads), iStackSize);

    return iNThreads;
  }
}  // namespace edm
