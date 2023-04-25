#ifndef FWCore_Concurrency_syncWait_h
#define FWCore_Concurrency_syncWait_h
//
//  syncWait.h
//
//  This file must be included before any other file that include tbb headers
//
//  Created by Chris Jones on 2/24/21.
//
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "oneapi/tbb/task_group.h"
#include <exception>

namespace edm {
  template <typename F>
  [[nodiscard]] std::exception_ptr syncWait(F&& iFunc) {
    std::exception_ptr exceptPtr{};
    oneapi::tbb::task_group group;
    FinalWaitingTask last{group};
    group.run([&]() { iFunc(WaitingTaskHolder(group, &last)); });  //group.run

    return last.waitNoThrow();
  }
}  // namespace edm
#endif /* FWCore_Concurrency_syncWait_h */
