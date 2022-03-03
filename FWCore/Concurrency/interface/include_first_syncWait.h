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
#include "oneapi/tbb/task_group.h"
#include "oneapi/tbb/task.h"
#include <exception>

namespace edm {
  template <typename F>
  [[nodiscard]] std::exception_ptr syncWait(F&& iFunc) {
    std::exception_ptr exceptPtr{};
    //oneapi::tbb::task::suspend can only be run from within a task running in this arena. For 1 thread,
    // it is often (always?) the case where not such task is being run here. Therefore we need
    // to use a temp task_group to start up such a task.
    oneapi::tbb::task_group group;
    group.run([&]() {
      oneapi::tbb::task::suspend([&](oneapi::tbb::task::suspend_point tag) {
        auto waitTask = make_waiting_task([tag, &exceptPtr](std::exception_ptr const* iExcept) {
          if (iExcept) {
            exceptPtr = *iExcept;
          }
          oneapi::tbb::task::resume(tag);
        });
        iFunc(WaitingTaskHolder(group, waitTask));
      });  //suspend
    });    //group.run

    group.wait();
    return exceptPtr;
  }
}  // namespace edm
#endif /* FWCore_Concurrency_syncWait_h */
