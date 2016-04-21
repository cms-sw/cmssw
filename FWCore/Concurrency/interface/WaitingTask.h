#ifndef FWCore_Concurrency_WaitingTask_h
#define FWCore_Concurrency_WaitingTask_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     WaitingTask
// 
/**\class WaitingTask WaitingTask.h FWCore/Concurrency/interface/WaitingTask.h

 Description: Task used by WaitingTaskList.

 Usage:
    Used as a callback to happen after a task has been completed. Includes the ability to hold an exception which has occurred while waiting.
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:46:31 CST 2013
// $Id$
//

// system include files
#include <atomic>
#include <exception>
#include <memory>
#include "tbb/task.h"

// user include files

// forward declarations

namespace edm {
  class WaitingTaskList;
  
  class WaitingTask : public tbb::task {
      
   public:
    friend class WaitingTaskList;
    
    ///Constructor
    WaitingTask() : m_ptr{nullptr} {}
    ~WaitingTask() override {
      delete m_ptr.load();
    };
      
    // ---------- const member functions ---------------------------
      
    ///Returns exception thrown by dependent task
    /** If the value is non-null then the dependent task failed.
    */
    std::exception_ptr const * exceptionPtr() const {
      return m_ptr.load();
    }
   private:
    
    ///Called if waited for task failed
    /**Allows transfer of the exception caused by the dependent task to be
     * moved to another thread.
     * This method should only be called by WaitingTaskList
     */
    void dependentTaskFailed(std::exception_ptr iPtr) {
      if (iPtr and not m_ptr) {
        auto temp = std::make_unique<std::exception_ptr>(iPtr);
        std::exception_ptr* expected = nullptr;
        if( m_ptr.compare_exchange_strong(expected, temp.get()) ) {
          temp.release();
        }
      }
    }
    
    std::atomic<std::exception_ptr*> m_ptr;
  };
  
}

#endif
