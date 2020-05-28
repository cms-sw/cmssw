#ifndef FWCore_Concurrency_WaitingTaskWithArenaSpawnHolder_h
#define FWCore_Concurrency_WaitingTaskWithArenaSpawnHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     WaitingTaskWithArenaSpawnHolder
//
/**\class edm::WaitingTaskWithArenaSpawnHolder

 Description: This holds a WaitingTask and can be passed to something
 the WaitingTask is waiting for. That allows that something to call
 doneWaiting to let the WaitingTask know it can run. The use of the
 arena allows one to call doneWaiting and do a spawn into that arena.
*/
//
// Original Author:  Chris Jones
//         Created:  26 May 2020
//

#include <exception>
#include <memory>

#include "tbb/task_arena.h"

namespace edm {
  class WaitingTask;
  class WaitingTaskHolder;

  class WaitingTaskWithArenaSpawnHolder {
  public:
    WaitingTaskWithArenaSpawnHolder();

    // Note that the arena will be the one containing the thread
    // that runs this constructor. This is the arena where you
    // eventually intend for the task to be spawned.
    explicit WaitingTaskWithArenaSpawnHolder(WaitingTask* iTask);

    ~WaitingTaskWithArenaSpawnHolder();

    WaitingTaskWithArenaSpawnHolder(WaitingTaskWithArenaSpawnHolder const& iHolder);

    WaitingTaskWithArenaSpawnHolder(WaitingTaskWithArenaSpawnHolder&& iOther);

    WaitingTaskWithArenaSpawnHolder& operator=(const WaitingTaskWithArenaSpawnHolder& iRHS);

    WaitingTaskWithArenaSpawnHolder& operator=(WaitingTaskWithArenaSpawnHolder&& iRHS);

    // This spawns the task. The arena is needed to get the task spawned
    // into the correct arena. doneWaiting must be called from a TBB controlled thread.
    void doneWaiting(std::exception_ptr iExcept);

  private:
    // ---------- member data --------------------------------
    WaitingTask* m_task;
    std::shared_ptr<tbb::task_arena> m_arena;
  };

  template <typename F>
  auto make_lambda_with_holder(WaitingTaskWithArenaSpawnHolder h, F&& f) {
    return [holder = std::move(h), func = std::forward<F>(f)]() mutable {
      try {
        func(holder);
      } catch (...) {
        holder.doneWaiting(std::current_exception());
      }
    };
  }

  template <typename ALLOC, typename F>
  auto make_waiting_task_with_holder(ALLOC&& iAlloc, WaitingTaskWithArenaSpawnHolder h, F&& f) {
    return make_waiting_task(
        std::forward<ALLOC>(iAlloc),
        [holder = h, func = make_lambda_with_holder(h, std::forward<F>(f))](std::exception_ptr const* excptr) mutable {
          if (excptr) {
            holder.doneWaiting(*excptr);
            return;
          }
          func();
        });
  }
}  // namespace edm
#endif
