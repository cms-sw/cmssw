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
    void push(oneapi::tbb::task_group& iGroup, T&& iAction);

    unsigned long outstandingTasks() const { return m_outstandingTasks; }
    std::size_t numberOfQueues() const { return m_queues.size(); }

  private:
    // ---------- member data --------------------------------
    std::vector<std::shared_ptr<SerialTaskQueue>> m_queues;
    std::atomic<unsigned long> m_outstandingTasks{0};

    template <typename T>
    void passDownChain(unsigned int iIndex, oneapi::tbb::task_group& iGroup, T&& iAction);

    template <typename T>
    void actionToRun(T&& iAction);
  };

  template <typename T>
  void SerialTaskQueueChain::push(oneapi::tbb::task_group& iGroup, T&& iAction) {
    ++m_outstandingTasks;
    if (m_queues.size() == 1) {
      m_queues[0]->push(iGroup, [this, iAction]() mutable { this->actionToRun(iAction); });
    } else {
      assert(!m_queues.empty());
      m_queues[0]->push(iGroup, [this, &iGroup, iAction]() mutable { this->passDownChain(1, iGroup, iAction); });
    }
  }

  template <typename T>
  void SerialTaskQueueChain::passDownChain(unsigned int iQueueIndex, oneapi::tbb::task_group& iGroup, T&& iAction) {
    //Have to be sure the queue associated to this running task
    // does not attempt to start another task
    m_queues[iQueueIndex - 1]->pause();
    //is this the last queue?
    if (iQueueIndex + 1 == m_queues.size()) {
      m_queues[iQueueIndex]->push(iGroup, [this, iAction]() mutable { this->actionToRun(iAction); });
    } else {
      auto nextQueue = iQueueIndex + 1;
      m_queues[iQueueIndex]->push(
          iGroup, [this, nextQueue, &iGroup, iAction]() mutable { this->passDownChain(nextQueue, iGroup, iAction); });
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
