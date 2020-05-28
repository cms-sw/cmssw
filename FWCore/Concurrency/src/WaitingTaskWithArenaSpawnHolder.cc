// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     WaitingTaskWithArenaSpawnHolder
//
// Original Author:  W. David Dagenhart
//         Created:  6 December 2017

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaSpawnHolder.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace edm {

  WaitingTaskWithArenaSpawnHolder::WaitingTaskWithArenaSpawnHolder() : m_task(nullptr) {}

  // Note that the arena will be the one containing the thread
  // that runs this constructor. This is the arena where you
  // eventually intend for the task to be spawned.
  WaitingTaskWithArenaSpawnHolder::WaitingTaskWithArenaSpawnHolder(WaitingTask* iTask)
      : m_task(iTask), m_arena(std::make_shared<tbb::task_arena>(tbb::task_arena::attach())) {
    m_task->increment_ref_count();
  }

  WaitingTaskWithArenaSpawnHolder::~WaitingTaskWithArenaSpawnHolder() {
    if (m_task) {
      doneWaiting(std::exception_ptr{});
    }
  }

  WaitingTaskWithArenaSpawnHolder::WaitingTaskWithArenaSpawnHolder(WaitingTaskWithArenaSpawnHolder const& iHolder)
      : m_task(iHolder.m_task), m_arena(iHolder.m_arena) {
    if (LIKELY(m_task != nullptr)) {
      m_task->increment_ref_count();
    }
  }

  WaitingTaskWithArenaSpawnHolder::WaitingTaskWithArenaSpawnHolder(WaitingTaskWithArenaSpawnHolder&& iOther)
      : m_task(iOther.m_task), m_arena(std::move(iOther.m_arena)) {
    iOther.m_task = nullptr;
  }

  WaitingTaskWithArenaSpawnHolder& WaitingTaskWithArenaSpawnHolder::operator=(
      const WaitingTaskWithArenaSpawnHolder& iRHS) {
    WaitingTaskWithArenaSpawnHolder tmp(iRHS);
    std::swap(m_task, tmp.m_task);
    std::swap(m_arena, tmp.m_arena);
    return *this;
  }

  WaitingTaskWithArenaSpawnHolder& WaitingTaskWithArenaSpawnHolder::operator=(WaitingTaskWithArenaSpawnHolder&& iRHS) {
    WaitingTaskWithArenaSpawnHolder tmp(std::move(iRHS));
    std::swap(m_task, tmp.m_task);
    std::swap(m_arena, tmp.m_arena);
    return *this;
  }

  void WaitingTaskWithArenaSpawnHolder::doneWaiting(std::exception_ptr iExcept) {
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
      m_arena->execute([task = task]() { tbb::task::spawn(*task); });
    }
  }
}  // namespace edm
