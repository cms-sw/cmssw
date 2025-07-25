#ifndef FWCore_Concurrency_SerialTaskQueue_h
#define FWCore_Concurrency_SerialTaskQueue_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     SerialTaskQueue
//
/**\class SerialTaskQueue SerialTaskQueue.h "FWCore/Concurrency/interface/SerialTaskQueue.h"

 Description: Runs only one task from the queue at a time

 Usage:
    A SerialTaskQueue is used to provide thread-safe access to a resource. You create a SerialTaskQueue
 for the resource. When every you need to perform an operation on the resource, you push a 'task' that
 does that operation onto the queue. The queue then makes sure to run one and only one task at a time.
 This guarantees serial access to the resource and therefore thread-safety.
 
    The 'tasks' managed by the SerialTaskQueue are just functor objects who which take no arguments and
 return no values. The simplest way to create a task is to use a C++11 lambda.
 
 Example: Imagine we have the following data structures.
 \code
 std::vector<int> values;
 edm::SerialTaskQueue queue;
 \endcode

 On thread 1 we can fill the vector
 \code
 for(int i=0; i<1000;++i) {
   queue.pushAndWait( [&values,i]{ values.push_back(i);} );
 }
 \endcode
 
 While on thread 2 we periodically print and stop when the vector is filled
 \code
 bool stop = false;
 while(not stop) {
   queue.pushAndWait([&false,&values] {
     if( 0 == (values.size() % 100) ) {
        std::cout <<values.size()<<std::endl;
     }
     if(values.size()>999) {
       stop = true;
     }
   });
 }
\endcode
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:14:39 CST 2013
// $Id$
//

// system include files
#include <atomic>
#include <cassert>

#include "oneapi/tbb/task_group.h"
#include "oneapi/tbb/concurrent_queue.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// user include files

// forward declarations
namespace edm {
  class SerialTaskQueue {
  public:
    SerialTaskQueue() : m_pauseCount{0}, m_taskChosen{false}, m_pickingNextTask{false} {}

    SerialTaskQueue(SerialTaskQueue&& iOther)
        : m_tasks(std::move(iOther.m_tasks)),
          m_pauseCount(iOther.m_pauseCount.exchange(0)),
          m_taskChosen(iOther.m_taskChosen.exchange(false)),
          m_pickingNextTask(false) {
      assert(m_tasks.empty() and m_taskChosen == false and iOther.m_pickingNextTask == false);
    }
    SerialTaskQueue(const SerialTaskQueue&) = delete;
    const SerialTaskQueue& operator=(const SerialTaskQueue&) = delete;

    ~SerialTaskQueue();

    // ---------- const member functions ---------------------
    /// Checks to see if the queue has been paused.
    /**\return true if the queue is paused
       * \sa pause(), resume()
       */
    bool isPaused() const { return m_pauseCount.load() != 0; }

    // ---------- member functions ---------------------------
    /// Pauses processing of additional tasks from the queue.
    /**
       * Any task already running will not be paused however once that
       * running task finishes no further tasks will be started.
       * Multiple calls to pause() are allowed, however each call to 
       * pause() must be balanced by a call to resume().
       * \return false if queue was already paused.
       * \sa resume(), isPaused()
       */
    bool pause() { return 1 == ++m_pauseCount; }

    /// Resumes processing if the queue was paused.
    /**
       * Multiple calls to resume() are allowed if there
       * were multiple calls to pause(). Only when we reach as
       * many resume() calls as pause() calls will the queue restart.
       * \return true if the call really restarts the queue
       * \sa pause(), isPaused()
       */
    bool resume();

    /// asynchronously pushes functor iAction into queue
    /**
       * The function will return immediately and iAction will either
       * process concurrently with the calling thread or wait until the
       * protected resource becomes available or until a CPU becomes available.
       * \param[in] iAction Must be a functor that takes no arguments and return no values.
       */
    template <typename T>
    void push(oneapi::tbb::task_group&, const T& iAction);

  private:
    /** Base class for all tasks held by the SerialTaskQueue */
    class TaskBase {
      friend class SerialTaskQueue;

      oneapi::tbb::task_group* group() { return m_group; }
      virtual void execute() = 0;

    public:
      virtual ~TaskBase() = default;

    protected:
      explicit TaskBase(oneapi::tbb::task_group* iGroup) : m_group(iGroup) {}

    private:
      oneapi::tbb::task_group* m_group;
    };

    template <typename T>
    class QueuedTask : public TaskBase {
    public:
      QueuedTask(oneapi::tbb::task_group& iGroup, const T& iAction) : TaskBase(&iGroup), m_action(iAction) {}

    private:
      void execute() final;

      T m_action;
    };

    friend class TaskBase;

    void pushTask(TaskBase*);
    TaskBase* pushAndGetNextTask(TaskBase*);
    TaskBase* finishedTask();
    //returns nullptr if a task is already being processed
    TaskBase* pickNextTask();

    void spawn(TaskBase&);

    // ---------- member data --------------------------------
    oneapi::tbb::concurrent_queue<TaskBase*> m_tasks;
    std::atomic<unsigned long> m_pauseCount;
    std::atomic<bool> m_taskChosen;
    std::atomic<bool> m_pickingNextTask;
  };

  template <typename T>
  void SerialTaskQueue::push(oneapi::tbb::task_group& iGroup, const T& iAction) {
    QueuedTask<T>* pTask{new QueuedTask<T>{iGroup, iAction}};
    pushTask(pTask);
  }

  template <typename T>
  void SerialTaskQueue::QueuedTask<T>::execute() {
    // Exception has to swallowed in order to avoid throwing from execute(). The user of SerialTaskQueue should handle exceptions within m_action().
    CMS_SA_ALLOW try { this->m_action(); } catch (...) {
    }
  }

}  // namespace edm

#endif
