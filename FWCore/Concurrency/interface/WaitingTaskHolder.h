#ifndef FWCore_Concurrency_WaitingTaskHolder_h
#define FWCore_Concurrency_WaitingTaskHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     WaitingTaskHolder
//
/**\class WaitingTaskHolder WaitingTaskHolder.h "WaitingTaskHolder.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  FWCore
//         Created:  Fri, 18 Nov 2016 20:30:42 GMT
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Concurrency/interface/WaitingTask.h"

// forward declarations

namespace edm {
  class WaitingTaskHolder {
  public:
    friend class WaitingTaskList;
    friend class WaitingTaskWithArenaHolder;

    WaitingTaskHolder() : m_task(nullptr) {}

    explicit WaitingTaskHolder(edm::WaitingTask* iTask) : m_task(iTask) { m_task->increment_ref_count(); }
    ~WaitingTaskHolder() {
      if (m_task) {
        doneWaiting(std::exception_ptr{});
      }
    }

    WaitingTaskHolder(const WaitingTaskHolder& iHolder) : m_task(iHolder.m_task) { m_task->increment_ref_count(); }

    WaitingTaskHolder(WaitingTaskHolder&& iOther) : m_task(iOther.m_task) { iOther.m_task = nullptr; }

    WaitingTaskHolder& operator=(const WaitingTaskHolder& iRHS) {
      WaitingTaskHolder tmp(iRHS);
      std::swap(m_task, tmp.m_task);
      return *this;
    }

    WaitingTaskHolder& operator=(WaitingTaskHolder&& iRHS) {
      WaitingTaskHolder tmp(std::move(iRHS));
      std::swap(m_task, tmp.m_task);
      return *this;
    }

    // ---------- const member functions ---------------------
    bool taskHasFailed() const { return m_task->exceptionPtr() != nullptr; }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

    /** Use in the case where you need to inform the parent task of a
     failure before some other child task which may be run later reports
     a different, but related failure. You must later call doneWaiting
     in the same thread passing the same exceptoin.
     */
    void presetTaskAsFailed(std::exception_ptr iExcept) {
      if (iExcept) {
        m_task->dependentTaskFailed(iExcept);
      }
    }

    void doneWaiting(std::exception_ptr iExcept) {
      if (iExcept) {
        m_task->dependentTaskFailed(iExcept);
      }
      //spawn can run the task before we finish
      // doneWaiting and some other thread might
      // try to reuse this object. Resetting
      // before spawn avoids problems
      auto task = m_task;
      m_task = nullptr;
      if (0 == task->decrement_ref_count()) {
        tbb::task::spawn(*task);
      }
    }

  private:
    WaitingTask* release_no_decrement() noexcept {
      auto t = m_task;
      m_task = nullptr;
      return t;
    }
    // ---------- member data --------------------------------
    WaitingTask* m_task;
  };
}  // namespace edm

#endif
