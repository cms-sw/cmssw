// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     WaitingTaskWithArenaHolder
//
// Original Author:  W. David Dagenhart
//         Created:  6 December 2017

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace edm {

  WaitingTaskWithArenaHolder::WaitingTaskWithArenaHolder() : m_task(nullptr) {}

  // Note that the arena will be the one containing the thread
  // that runs this constructor. This is the arena where you
  // eventually intend for the task to be spawned.
  WaitingTaskWithArenaHolder::WaitingTaskWithArenaHolder(WaitingTask* iTask)
      : m_task(iTask), m_arena(std::make_shared<tbb::task_arena>(tbb::task_arena::attach())) {
    m_task->increment_ref_count();
  }

  WaitingTaskWithArenaHolder::WaitingTaskWithArenaHolder(WaitingTaskHolder&& iTask)
      : m_task(iTask.release_no_decrement()), m_arena(std::make_shared<tbb::task_arena>(tbb::task_arena::attach())) {}

  WaitingTaskWithArenaHolder::~WaitingTaskWithArenaHolder() {
    if (m_task) {
      doneWaiting(std::exception_ptr{});
    }
  }

  WaitingTaskWithArenaHolder::WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder const& iHolder)
      : m_task(iHolder.m_task), m_arena(iHolder.m_arena) {
    if (LIKELY(m_task != nullptr)) {
      m_task->increment_ref_count();
    }
  }

  WaitingTaskWithArenaHolder::WaitingTaskWithArenaHolder(WaitingTaskWithArenaHolder&& iOther)
      : m_task(iOther.m_task), m_arena(std::move(iOther.m_arena)) {
    iOther.m_task = nullptr;
  }

  WaitingTaskWithArenaHolder& WaitingTaskWithArenaHolder::operator=(const WaitingTaskWithArenaHolder& iRHS) {
    WaitingTaskWithArenaHolder tmp(iRHS);
    std::swap(m_task, tmp.m_task);
    std::swap(m_arena, tmp.m_arena);
    return *this;
  }

  WaitingTaskWithArenaHolder& WaitingTaskWithArenaHolder::operator=(WaitingTaskWithArenaHolder&& iRHS) {
    WaitingTaskWithArenaHolder tmp(std::move(iRHS));
    std::swap(m_task, tmp.m_task);
    std::swap(m_arena, tmp.m_arena);
    return *this;
  }

  // This spawns the task. The arena is needed to get the task spawned
  // into the correct arena of threads. Use of the arena allows doneWaiting
  // to be called from a thread outside the arena of threads that will manage
  // the task. doneWaiting can be called from a non-TBB thread.
  void WaitingTaskWithArenaHolder::doneWaiting(std::exception_ptr iExcept) {
    if (iExcept) {
      m_task->dependentTaskFailed(iExcept);
    }
    //enqueue can run the task before we finish
    // doneWaiting and some other thread might
    // try to reuse this object. Resetting
    // before enqueue avoids problems
    auto task = m_task;
    m_task = nullptr;
    if (0 == task->decrement_ref_count()) {
      // The enqueue call will cause a worker thread to be created in
      // the arena if there is not one already.
      m_arena->enqueue([task = task]() { tbb::task::spawn(*task); });
    }
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

  WaitingTaskHolder WaitingTaskWithArenaHolder::makeWaitingTaskHolderAndRelease() {
    WaitingTaskHolder holder(m_task);
    m_task->decrement_ref_count();
    m_task = nullptr;
    return holder;
  }
}  // namespace edm
