#ifndef FWCore_Concurrency_WaitingTaskWithArenaHolder_h
#define FWCore_Concurrency_WaitingTaskWithArenaHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     WaitingTaskWithArenaHolder
//
/**\class edm::WaitingTaskWithArenaHolder

 Description: This holds a WaitingTask and can be passed to something
 the WaitingTask is waiting for. That allows that something to call
 doneWaiting to let the WaitingTask know it can run. The use of the
 arena allows one to call doneWaiting from a thread external to the
 arena where the task should run. The external thread might be a non-TBB
 thread.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  9 November 2017
//

#include <exception>
#include <memory>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm {
  class WaitingTask;
  class WaitingTaskHolder;

  class WaitingTaskWithArenaHolder {
  public:
    WaitingTaskWithArenaHolder();

    // Note that the arena will be the one containing the thread
    // that runs this constructor. This is the arena where you
    // eventually intend for the task to be spawned.
    explicit WaitingTaskWithArenaHolder(oneapi::tbb::task_group&, WaitingTask* iTask);

    // Takes ownership of the underlying task and uses the current
    // arena.
    explicit WaitingTaskWithArenaHolder(WaitingTaskHolder&& iTask);

    ~WaitingTaskWithArenaHolder();

    WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder const& iHolder);

    WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder&& iOther);

    WaitingTaskWithArenaHolder& operator=(const WaitingTaskWithArenaHolder& iRHS);

    WaitingTaskWithArenaHolder& operator=(WaitingTaskWithArenaHolder&& iRHS);

    // This spawns the task. The arena is needed to get the task spawned
    // into the correct arena of threads. Use of the arena allows doneWaiting
    // to be called from a thread outside the arena of threads that will manage
    // the task. doneWaiting can be called from a non-TBB thread.
    void doneWaiting(std::exception_ptr iExcept);

    // This next function is useful if you know from the context that
    // m_arena (which is set when the constructor was executes) is the
    // same arena in which you want to execute the doneWaiting function.
    // It allows an optimization which avoids the enqueue step in the
    // doneWaiting function.
    //
    // Be warned though that in general this function cannot be used.
    // Spawning a task outside the correct arena could create a new separate
    // arena with its own extra TBB worker threads if this function is used
    // in an inappropriate context (and silently such that you might not notice
    // the problem quickly).
    WaitingTaskHolder makeWaitingTaskHolderAndRelease();

    bool taskHasFailed() const noexcept;

    bool hasTask() const noexcept;

    /** since oneapi::tbb::task_group is thread safe, we can return it non-const from here since
        the object is not really part of the state of the holder
     */
    CMS_SA_ALLOW oneapi::tbb::task_group* group() const { return m_group; }

  private:
    // ---------- member data --------------------------------
    WaitingTask* m_task;
    oneapi::tbb::task_group* m_group;
    std::shared_ptr<oneapi::tbb::task_arena> m_arena;
  };

  template <typename F>
  auto make_lambda_with_holder(WaitingTaskWithArenaHolder h, F&& f) {
    return [holder = std::move(h), func = std::forward<F>(f)]() mutable {
      try {
        func(holder);
      } catch (...) {
        holder.doneWaiting(std::current_exception());
      }
    };
  }

  template <typename F>
  auto make_waiting_task_with_holder(WaitingTaskWithArenaHolder h, F&& f) {
    return make_waiting_task(
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
