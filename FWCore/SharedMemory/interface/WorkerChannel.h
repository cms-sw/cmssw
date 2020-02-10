#ifndef FWCore_SharedMemory_WorkerChannel_h
#define FWCore_SharedMemory_WorkerChannel_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     WorkerChannel
//
/**\class WorkerChannel WorkerChannel.h " FWCore/SharedMemory/interface/WorkerChannel.h"

 Description:  Primary communication channel for the Worker process

 Usage:
    Used in conjunction with the ControllerChannel

*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <string>
#include "boost/interprocess/managed_shared_memory.hpp"
#include "boost/interprocess/sync/named_mutex.hpp"
#include "boost/interprocess/sync/named_condition.hpp"
#include "boost/interprocess/sync/scoped_lock.hpp"

// user include files
#include "FWCore/Utilities/interface/Transition.h"

// forward declarations

namespace edm::shared_memory {
  class WorkerChannel {
  public:
    /** iName must match the value from ControllerChannel::sharedMemoryName()
     iUniqueName must match the value from ControllerChannel::uniqueID()     
     */
    WorkerChannel(std::string const& iName, const std::string& iUniqueID);
    WorkerChannel(const WorkerChannel&) = delete;
    const WorkerChannel& operator=(const WorkerChannel&) = delete;
    WorkerChannel(WorkerChannel&&) = delete;
    const WorkerChannel& operator=(WorkerChannel&&) = delete;

    // ---------- member functions ---------------------------
    /// the lock is made accessible so that the WorkerMonitorThread can be used to unlock it in the event of a unix signal
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex>* accessLock() { return &lock_; }

    ///This can be used with ReadBuffer to keep Controller and Worker in sync
    char* toWorkerBufferIndex() { return toWorkerBufferIndex_; }
    ///This can be used with WriteBuffer to keep Controller and Worker in sync
    char* fromWorkerBufferIndex() { return fromWorkerBufferIndex_; }

    ///Matches the ControllerChannel::setupWorker call
    void workerSetupDone() {
      //The controller is waiting for the worker to be setup
      notifyController();
    }

    /**Matches the ControllerChannel::doTransition calls.
     iF is a function that takes as arguments a edm::Transition and unsigned long long
     */
    template <typename F>
    void handleTransitions(F&& iF) {
      while (true) {
        waitForController();
        if (stopRequested()) {
          break;
        }

        iF(transition(), transitionID());
        notifyController();
      }
    }

    ///call this from the `handleTransitions` functor
    void shouldKeepEvent(bool iChoice) { *keepEvent_ = iChoice; }

    ///These are here for expert use
    void notifyController() { cndToController_.notify_all(); }
    void waitForController() { cndFromController_.wait(lock_); }

    // ---------- const member functions ---------------------------
    edm::Transition transition() const noexcept { return *transitionType_; }
    unsigned long long transitionID() const noexcept { return *transitionID_; }
    bool stopRequested() const noexcept { return *stop_; }

  private:
    // ---------- member data --------------------------------
    boost::interprocess::managed_shared_memory managed_shm_;

    boost::interprocess::named_mutex mutex_;
    boost::interprocess::named_condition cndFromController_;
    bool* stop_;
    edm::Transition* transitionType_;
    unsigned long long* transitionID_;
    char* toWorkerBufferIndex_;
    char* fromWorkerBufferIndex_;
    boost::interprocess::named_condition cndToController_;
    bool* keepEvent_;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock_;
  };
}  // namespace edm::shared_memory

#endif
