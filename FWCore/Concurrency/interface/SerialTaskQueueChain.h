#ifndef FWCore_Concurrency_SerialTaskQueueChain_h
#define FWCore_Concurrency_SerialTaskQueueChain_h
// -*- C++ -*-
//
// Package:     FWCore/Concurrency
// Class  :     SerialTaskQueueChain
//
/**\class SerialTaskQueueChain SerialTaskQueueChain.h "SerialTaskQueueChain.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Mon, 15 Aug 2016 18:04:02 GMT
//

// system include files
#include <cassert>
#include <memory>
#include <vector>

// user include files
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
namespace edm {
  class SerialTaskQueueChain {
  public:
    SerialTaskQueueChain() {}
    explicit SerialTaskQueueChain(std::vector<std::shared_ptr<SerialTaskQueue>> iQueues)
        : m_queues(std::move(iQueues)) {}

    SerialTaskQueueChain(const SerialTaskQueueChain&) = delete;
    SerialTaskQueueChain& operator=(const SerialTaskQueueChain&) = delete;
    SerialTaskQueueChain(SerialTaskQueueChain&& iOld)
        : m_queues(std::move(iOld.m_queues)), m_outstandingTasks{iOld.m_outstandingTasks.load()} {}

    SerialTaskQueueChain& operator=(SerialTaskQueueChain&& iOld) {
      m_queues = std::move(iOld.m_queues);
      m_outstandingTasks.store(iOld.m_outstandingTasks.load());
      return *this;
    }

    /// asynchronously pushes functor iAction into queue
    /**
     * The function will return immediately and iAction will either
     * process concurrently with the calling thread or wait until the
     * protected resource becomes available or until a CPU becomes available.
     * \param[in] iAction Must be a functor that takes no arguments and return no values.
     */
    template <typename T>
    void push(T&& iAction);

    /// synchronously pushes functor iAction into queue
    /**
     * The function will wait until iAction has completed before returning.
     * If another task is already running on the queue, the system is allowed
     * to find another TBB task to execute while waiting for the iAction to finish.
     * In that way the core is not idled while waiting.
     * \param[in] iAction Must be a functor that takes no arguments and return no values.
     */
    template <typename T>
    void pushAndWait(T&& iAction);

    unsigned long outstandingTasks() const { return m_outstandingTasks; }
    std::size_t numberOfQueues() const { return m_queues.size(); }

  private:
    // ---------- member data --------------------------------
    std::vector<std::shared_ptr<SerialTaskQueue>> m_queues;
    std::atomic<unsigned long> m_outstandingTasks{0};

    template <typename T>
    void passDownChain(unsigned int iIndex, T&& iAction);

    template <typename T>
    void actionToRun(T&& iAction);
  };

  template <typename T>
  void SerialTaskQueueChain::push(T&& iAction) {
    ++m_outstandingTasks;
    if (m_queues.size() == 1) {
      m_queues[0]->push([this, iAction]() mutable { this->actionToRun(iAction); });
    } else {
      assert(!m_queues.empty());
      m_queues[0]->push([this, iAction]() mutable { this->passDownChain(1, iAction); });
    }
  }

  template <typename T>
  void SerialTaskQueueChain::pushAndWait(T&& iAction) {
    auto destry = [](tbb::task* iTask) { tbb::task::destroy(*iTask); };

    std::unique_ptr<tbb::task, decltype(destry)> waitTask(new (tbb::task::allocate_root()) tbb::empty_task, destry);
    waitTask->set_ref_count(3);

    std::exception_ptr ptr;
    auto waitTaskPtr = waitTask.get();
    push([waitTaskPtr, iAction, &ptr]() {
      //must wait until exception ptr would be set
      auto dec = [](tbb::task* iTask) { iTask->decrement_ref_count(); };
      std::unique_ptr<tbb::task, decltype(dec)> sentry(waitTaskPtr, dec);
      // Caught exception is rethrown further below.
      CMS_SA_ALLOW try { iAction(); } catch (...) {
        ptr = std::current_exception();
      }
    });

    waitTask->decrement_ref_count();
    waitTask->wait_for_all();

    if (ptr) {
      std::rethrow_exception(ptr);
    }
  }

  template <typename T>
  void SerialTaskQueueChain::passDownChain(unsigned int iQueueIndex, T&& iAction) {
    //Have to be sure the queue associated to this running task
    // does not attempt to start another task
    m_queues[iQueueIndex - 1]->pause();
    //is this the last queue?
    if (iQueueIndex + 1 == m_queues.size()) {
      m_queues[iQueueIndex]->push([this, iAction]() mutable { this->actionToRun(iAction); });
    } else {
      auto nextQueue = iQueueIndex + 1;
      m_queues[iQueueIndex]->push([this, nextQueue, iAction]() mutable { this->passDownChain(nextQueue, iAction); });
    }
  }

  template <typename T>
  void SerialTaskQueueChain::actionToRun(T&& iAction) {
    //even if an exception happens we will resume the queues.
    using Queues = std::vector<std::shared_ptr<SerialTaskQueue>>;
    auto sentryAction = [](SerialTaskQueueChain* iChain) {
      auto& vec = iChain->m_queues;
      for (auto it = vec.rbegin() + 1; it != vec.rend(); ++it) {
        (*it)->resume();
      }
      --(iChain->m_outstandingTasks);
    };

    std::unique_ptr<SerialTaskQueueChain, decltype(sentryAction)> sentry(this, sentryAction);
    iAction();
  }
}  // namespace edm

#endif
