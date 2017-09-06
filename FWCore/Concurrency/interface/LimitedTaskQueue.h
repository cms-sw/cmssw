#ifndef FWCore_Concurrency_LimitedTaskQueue_h
#define FWCore_Concurrency_LimitedTaskQueue_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     LimitedTaskQueue
// 
/**\class LimitedTaskQueue LimitedTaskQueue.h "FWCore/Concurrency/interface/LimitedTaskQueue.h"

 Description: Runs a set number of tasks from the queue at a time

 Usage:
    A LimitedTaskQueue is used to provide access to a limited thread-safe resource. You create a LimitedTaskQueue
 for the resource. When every you need to perform an operation on the resource, you push a 'task' that
 does that operation onto the queue. The queue then makes sure to run a limited number of tasks at a time.
 
    The 'tasks' managed by the LimitedTaskQueue are just functor objects who which take no arguments and
 return no values. The simplest way to create a task is to use a C++11 lambda.
 
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:14:39 CST 2013
// $Id$
//

// system include files
#include <atomic>
#include <vector>
#include <memory>

#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

// user include files

// forward declarations
namespace edm {
   class LimitedTaskQueue
   {
   public:
     LimitedTaskQueue(unsigned int iLimit):
     m_queues{iLimit}
      {  }
      
     
      // ---------- member functions ---------------------------
     
      /// asynchronously pushes functor iAction into queue
      /**
       * The function will return immediately and iAction will either
       * process concurrently with the calling thread or wait until the
       * protected resource becomes available or until a CPU becomes available.
       * \param[in] iAction Must be a functor that takes no arguments and return no values.
       */
      template<typename T>
      void push(const T& iAction);
      
      /// synchronously pushes functor iAction into queue
      /**
       * The function will wait until iAction has completed before returning.
       * If another task is already running on the queue, the system is allowed
       * to find another TBB task to execute while waiting for the iAction to finish.
       * In that way the core is not idled while waiting.
       * \param[in] iAction Must be a functor that takes no arguments and return no values.
       */
      template<typename T>
      void pushAndWait(const T& iAction);
     
     unsigned int concurrencyLimit() const { return m_queues.size(); }
   private:
      LimitedTaskQueue(const LimitedTaskQueue&) = delete;
      const LimitedTaskQueue& operator=(const LimitedTaskQueue&) = delete;
     
      // ---------- member data --------------------------------
     std::vector<SerialTaskQueue> m_queues;
   };
   
   template<typename T>
   void LimitedTaskQueue::push(const T& iAction) {
     auto set_to_run = std::make_shared<std::atomic<bool>>(false);
     for(auto& q: m_queues) {
       q.push([set_to_run,iAction]() {
         bool expected = false;
         if(set_to_run->compare_exchange_strong(expected,true)) {
           iAction();
         }
       });
     }
   }
   
   template<typename T>
   void LimitedTaskQueue::pushAndWait(const T& iAction) {
      tbb::empty_task* waitTask = new (tbb::task::allocate_root()) tbb::empty_task;
      waitTask->set_ref_count(2);
     auto set_to_run = std::make_shared<std::atomic<bool>>(false);
     for(auto& q: m_queues) {
       q.push([set_to_run,waitTask,iAction]() {
         bool expected = false;
         if(set_to_run->compare_exchange_strong(expected,true)) {
           try {
             iAction();
           }catch(...) {}
           waitTask->decrement_ref_count();
         }
       });
     }
     waitTask->wait_for_all();
     tbb::task::destroy(*waitTask);
   }
   
}

#endif
