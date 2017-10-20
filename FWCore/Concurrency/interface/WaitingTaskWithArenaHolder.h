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
 arena allows one to call doneWaiting from a thread external to arena
 where the task should run. The external thread might be a non-TBB
 thread.

 Usage:

*/
//
// Original Author:  W. David Dagenhart
//         Created:  9 November 2017
//

#include <memory>

#include "tbb/task_arena.h"

#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

namespace edm {

  class WaitingTaskWithArenaHolder {
  public:

    WaitingTaskWithArenaHolder() : m_task(nullptr) {}

    // Note that the arena is to be the one containing the thread
    // that runs this constructor. This is the arena where you
    // eventually intend for the task to be spawned.
    explicit WaitingTaskWithArenaHolder(edm::WaitingTask* iTask) :
      m_task(iTask),
      m_arena(std::make_shared<tbb::task_arena>(tbb::task_arena::attach())) {

      m_task->increment_ref_count();
    }

    ~WaitingTaskWithArenaHolder() {
      if(m_task) {
        doneWaiting(std::exception_ptr{});
      }
    }

    WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder const& iHolder) :
      m_task(iHolder.m_task),
      m_arena(iHolder.m_arena) {

      m_task->increment_ref_count();
    }

    WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder&& iOther) :
      m_task(iOther.m_task),
      m_arena(std::move(iOther.m_arena)) {

      iOther.m_task = nullptr;
    }

    WaitingTaskWithArenaHolder& operator=(const WaitingTaskWithArenaHolder& iRHS) {
      WaitingTaskWithArenaHolder tmp(iRHS);
      std::swap(m_task, tmp.m_task);
      std::swap(m_arena, tmp.m_arena);
      return *this;
    }

    // This spawns the task. The arena is needed to get the task spawned
    // into the correct arena of threads. Use of the arena allows doneWaiting
    // to be called from a thread outside the arena of threads that will manage
    // the task. doneWaiting can be called from a non-TBB thread.
    void doneWaiting(std::exception_ptr iExcept) {
      if(iExcept) {
        m_task->dependentTaskFailed(iExcept);
      }
      if(0 == m_task->decrement_ref_count()) {
        // The enqueue call will cause a worker thread to be created in
        // the arena if there is not one already.
        m_arena->enqueue( [m_task = m_task](){ tbb::task::spawn(*m_task); });
      }
      m_task = nullptr;
    }

    // This next function is useful if you know from the context that
    // m_arena (which is set when the  constructor was executes) is the
    // same arena in which you want to execute the doneWaiting function.
    // It allows an optimization which avoids the enqueue step in the
    // doneWaiting function.
    //
    // Be warned though that in general this function cannot be used.
    // Spawning a task outside the correct arena could create a new separate
    // arena with its own extra TBB worker threads if this function is used
    // in an inappropriate context (and silently such that you might not notice
    // the problem quickly).

    WaitingTaskHolder makeWaitingTaskHolderAndRelease() {
      WaitingTaskHolder holder(m_task);
      m_task->decrement_ref_count();
      m_task = nullptr;
      return holder;
    }

  private:

    // ---------- member data --------------------------------
    WaitingTask* m_task;
    std::shared_ptr<tbb::task_arena> m_arena;
  };
}
#endif
