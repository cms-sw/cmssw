#ifndef FWCore_SharedMemory_ControllerChannel_h
#define FWCore_SharedMemory_ControllerChannel_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     ControllerChannel
//
/**\class ControllerChannel ControllerChannel.h " FWCore/SharedMemory/interface/ControllerChannel.h"

 Description: Primary communication channel for the Controller process

 Usage:
    Works in conjunction with the WorkerChannel

*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <string>
#include <iostream>
#include "boost/interprocess/managed_shared_memory.hpp"
#include "boost/interprocess/sync/named_mutex.hpp"
#include "boost/interprocess/sync/named_condition.hpp"
#include "boost/interprocess/sync/scoped_lock.hpp"

// user include files
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations

namespace edm::shared_memory {
  class ControllerChannel {
  public:
    /** iName is used as the base for the shared memory name. The full name uses iID as well as getpid() to create the value sharedMemoryName().
     iID allows multiple ControllChannels to use the same base name iName.
     */
    ControllerChannel(std::string const& iName, int iID);
    ~ControllerChannel();
    ControllerChannel(const ControllerChannel&) = delete;
    const ControllerChannel& operator=(const ControllerChannel&) = delete;
    ControllerChannel(ControllerChannel&&) = delete;
    const ControllerChannel& operator=(ControllerChannel&&) = delete;

    // ---------- member functions ---------------------------

    /** setupWorker must be called only once and done before any calls to doTransition. The functor iF should setup values associated
     with shared memory use, such as manipulating the value from toWorkerBufferIndex(). The call to setupWorker proper synchronizes
     the Controller and Worker processes.
     */
    template <typename F>
    void setupWorker(F&& iF) {
      using namespace boost::interprocess;
      scoped_lock<named_mutex> lock(mutex_);
      iF();
      using namespace boost::posix_time;
      //std::cout << id_ << " waiting for external process" << std::endl;

      if (not cndToMain_.timed_wait(lock, microsec_clock::universal_time() + seconds(60))) {
        //std::cout << id_ << " FAILED waiting for external process" << std::endl;
        throw cms::Exception("ExternalFailed");
      } else {
        //std::cout << id_ << " done waiting for external process" << std::endl;
      }
    }

    template <typename F>
    bool doTransition(F&& iF, edm::Transition iTrans, unsigned long long iTransitionID) {
      using namespace boost::interprocess;

      //std::cout << id_ << " taking from lock" << std::endl;
      scoped_lock<named_mutex> lock(mutex_);

      if (not wait(lock, iTrans, iTransitionID)) {
        return false;
      }
      //std::cout <<id_<<"running doTranstion command"<<std::endl;
      iF();
      return true;
    }

    ///This can be used with WriteBuffer to keep Controller and Worker in sync
    char* toWorkerBufferIndex() { return toWorkerBufferIndex_; }
    ///This can be used with ReadBuffer to keep Controller and Worker in sync
    char* fromWorkerBufferIndex() { return fromWorkerBufferIndex_; }

    void stopWorker() {
      //std::cout <<"stopWorker"<<std::endl;
      using namespace boost::interprocess;
      scoped_lock<named_mutex> lock(mutex_);
      *stop_ = true;
      //std::cout <<"stopWorker sending notification"<<std::endl;
      cndFromMain_.notify_all();
    }

    // ---------- const member functions ---------------------------
    std::string const& sharedMemoryName() const { return smName_; }
    std::string uniqueID() const { return uniqueName(""); }

    //should only be called after calling `doTransition`
    bool shouldKeepEvent() const { return *keepEvent_; }

  private:
    static char* bufferIndex(const char* iWhich, boost::interprocess::managed_shared_memory& mem);

    std::string uniqueName(std::string iBase) const;

    bool wait(boost::interprocess::scoped_lock<boost::interprocess::named_mutex>& lock,
              edm::Transition iTrans,
              unsigned long long iTransID);

    // ---------- member data --------------------------------
    int id_;
    std::string smName_;
    boost::interprocess::managed_shared_memory managed_sm_;
    char* toWorkerBufferIndex_;
    char* fromWorkerBufferIndex_;

    boost::interprocess::named_mutex mutex_;
    boost::interprocess::named_condition cndFromMain_;

    boost::interprocess::named_condition cndToMain_;

    edm::Transition* transitionType_;
    unsigned long long* transitionID_;
    bool* stop_;
    bool* keepEvent_;
  };
}  // namespace edm::shared_memory

#endif
