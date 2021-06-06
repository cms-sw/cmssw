#ifndef FWCore_Concurrency_TaskBase_h
#define FWCore_Concurrency_TaskBase_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     TaskBase
//
/**\class TaskBase TaskBase.h FWCore/Concurrency/interface/TaskBase.h

 Description: Base class for tasks.

 Usage:
    Used as a callback to happen after a task has been completed.
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 5 13:46:31 CST 2020
// $Id$
//

// system include files
#include <atomic>
#include <exception>
#include <memory>

// user include files

// forward declarations

namespace edm {
  class TaskBase {
  public:
    friend class TaskSentry;

    ///Constructor
    TaskBase() : m_refCount{0} {}
    virtual ~TaskBase() = default;

    virtual void execute() = 0;

    void increment_ref_count() { ++m_refCount; }
    unsigned int decrement_ref_count() { return --m_refCount; }

  private:
    virtual void recycle() { delete this; }

    std::atomic<unsigned int> m_refCount{0};
  };

  class TaskSentry {
  public:
    TaskSentry(TaskBase* iTask) : m_task{iTask} {}
    ~TaskSentry() { m_task->recycle(); }
    TaskSentry() = delete;
    TaskSentry(TaskSentry const&) = delete;
    TaskSentry(TaskSentry&&) = delete;
    TaskSentry operator=(TaskSentry const&) = delete;
    TaskSentry operator=(TaskSentry&&) = delete;

  private:
    TaskBase* m_task;
  };
}  // namespace edm

#endif
