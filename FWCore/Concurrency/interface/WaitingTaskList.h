#ifndef FWCore_Concurrency_WaitingTaskList_h
#define FWCore_Concurrency_WaitingTaskList_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     WaitingTaskList
//
/**\class WaitingTaskList WaitingTaskList.h FWCore/Concurrency/interface/WaitingTaskList.h

 Description: Handles starting tasks once some resource becomes available.

 Usage:
    This class can be used to have tasks wait to be spawned until a resource is available.
 Tasks that want to use the resource are added to the list by calling add(tbb::task*).
 When the resource becomes available one calls doneWaiting() and then any waiting tasks will
 be spawned. If a call to add() is made after doneWaiting() the newly added task will
 immediately be spawned.
 The class can be reused by calling reset(). However, reset() is not thread-safe so one 
 must be certain neither add(...) nor doneWaiting() is called while reset() is running.
 
 An example usage would be if you had a task doing a long calculation (the resource) and
 then several other tasks have been created in a different thread and before running those
 new tasks you need the result of the long calculation.
 \code
 class CalcTask : public edm::WaitingTask {
    public:
    CalcTask(edm::WaitingTaskList* iWL, Value* v):
    m_waitList(iWL), m_output(v) {}
 
    tbb::task* execute() {
     std::exception_ptr ptr;
     try {
       *m_output = doCalculation();
     } catch(...) {
       ptr = std::current_exception();
     }
     m_waitList.doneWaiting(ptr);
     return nullptr;
    }
    private:
     edm::WaitingTaskList* m_waitList;
     Value* m_output;
 };
 \endcode
 
 In one part of the code we can setup the shared resource
 \code
 WaitingTaskList waitList;
 Value v;
 \endcode

 In another part we can start the calculation
 \code
 tbb::task* calc = new(tbb::task::allocate_root()) CalcTask(&waitList,&v);
 tbb::task::spawn(calc);
 \endcode
 
 Finally in some unrelated part of the code we can create tasks that need the calculation
 \code
 tbb::task* t1 = makeTask1(v);
 waitList.add(t1);
 tbb::task* t2 = makeTask2(v);
 waitList.add(t2);
 \endcode

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:46:31 CST 2013
// $Id$
//

// system include files
#include <atomic>

// user include files
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations

namespace edm {
  class EmptyWaitingTask : public WaitingTask {
  public:
    EmptyWaitingTask() = default;

    tbb::task* execute() override { return nullptr; }
  };

  namespace waitingtask {
    struct TaskDestroyer {
      void operator()(tbb::task* iTask) const { tbb::task::destroy(*iTask); }
    };
  }  // namespace waitingtask
  ///Create an EmptyWaitingTask which will properly be destroyed
  inline std::unique_ptr<edm::EmptyWaitingTask, waitingtask::TaskDestroyer> make_empty_waiting_task() {
    return std::unique_ptr<edm::EmptyWaitingTask, waitingtask::TaskDestroyer>(new (tbb::task::allocate_root())
                                                                                  edm::EmptyWaitingTask{});
  }

  class WaitingTaskList {
  public:
    ///Constructor
    /**The WaitingTaskList is initial set to waiting.
       * \param[in] iInitialSize specifies the initial size of the cache used to hold waiting tasks.
       * The value is only useful for optimization as the object can resize itself.
       */
    explicit WaitingTaskList(unsigned int iInitialSize = 2);
    WaitingTaskList(const WaitingTaskList&) = delete;                   // stop default
    const WaitingTaskList& operator=(const WaitingTaskList&) = delete;  // stop default
    ~WaitingTaskList() = default;

    // ---------- member functions ---------------------------

    /** Use in the case where you need to inform the parent task of a
       failure before some other child task which may be run later reports
       a different, but related failure. You must later call doneWaiting
       with same exception later in the same thread.
       */
    void presetTaskAsFailed(std::exception_ptr iExcept);

    ///Adds task to the waiting list
    /**If doneWaiting() has already been called then the added task will immediately be spawned.
       * If that is not the case then the task will be held until doneWaiting() is called and will
       * then be spawned.
       * Calls to add() and doneWaiting() can safely be done concurrently.
       */
    void add(WaitingTask*);

    ///Signals that the resource is now available and tasks should be spawned
    /**The owner of the resource calls this function to allow the waiting tasks to
       * start accessing it.
       * If the task fails, a non 'null' std::exception_ptr should be used.
       * To have tasks wait again one must call reset().
       * Calls to add() and doneWaiting() can safely be done concurrently.
       */
    void doneWaiting(std::exception_ptr iPtr);

    ///Resets access to the resource so that added tasks will wait.
    /**The owner of the resouce calls reset() to make tasks wait.
       * Calling reset() is NOT thread safe. The system must guarantee that no tasks are
       * using the resource when reset() is called and neither add() nor doneWaiting() can
       * be called concurrently with reset().
       */
    void reset();

  private:
    /**Handles spawning the tasks,
       * safe to call from multiple threads
       */
    void announce();

    struct WaitNode {
      WaitingTask* m_task;
      std::atomic<WaitNode*> m_next;
      bool m_fromCache;

      void setNextNode(WaitNode* iNext) { m_next = iNext; }

      WaitNode* nextNode() const { return m_next; }
    };

    WaitNode* createNode(WaitingTask* iTask);

    // ---------- member data --------------------------------
    std::atomic<WaitNode*> m_head;
    std::unique_ptr<WaitNode[]> m_nodeCache;
    CMS_THREAD_GUARD(m_waiting) std::exception_ptr m_exceptionPtr;
    unsigned int m_nodeCacheSize;
    std::atomic<unsigned int> m_lastAssignedCacheIndex;
    std::atomic<bool> m_waiting;
  };
}  // namespace edm

#endif
