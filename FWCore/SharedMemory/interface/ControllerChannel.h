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
#include "boost/date_time/posix_time/posix_time_types.hpp"

// user include files
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/SharedMemory/interface/BufferInfo.h"

// forward declarations

namespace edm::shared_memory {
  class ControllerChannel {
  public:
    /** iName is used as the base for the shared memory name. The full name uses iID as well as getpid() to create the value sharedMemoryName().
     iID allows multiple ControllChannels to use the same base name iName.
     */
    ControllerChannel(std::string const& iName, int iID, unsigned int iMaxWaitInSeconds);
    ~ControllerChannel();
    ControllerChannel(const ControllerChannel&) = delete;
    const ControllerChannel& operator=(const ControllerChannel&) = delete;
    ControllerChannel(ControllerChannel&&) = delete;
    const ControllerChannel& operator=(ControllerChannel&&) = delete;

    // ---------- member functions ---------------------------

    /** setupWorker must be called only once and done before any calls to doTransition. The functor iF should setup values associated
     with shared memory use, such as manipulating the value from toWorkerBufferInfo(). The call to setupWorker proper synchronizes
     the Controller and Worker processes.
     */
    template <typename F>
    void setupWorker(F&& iF) {
      using namespace boost::interprocess;
      scoped_lock<named_mutex> lock(mutex_);
      iF();
      using namespace boost::posix_time;
      //std::cout << id_ << " waiting for external process" << std::endl;
      if (not wait(lock)) {
        //std::cout << id_ << " FAILED waiting for external process" << std::endl;
        *stop_ = true;
        throw edm::Exception(edm::errors::ExternalFailure)
            << "Failed waiting for external process while setting up the process. Timed out after " << maxWaitInSeconds_
            << " seconds.";
      } else {
        //std::cout << id_ << " done waiting for external process" << std::endl;
      }
    }

    /** setupWorkerWithRetry works just like setupWorker except it gives a way to continue waiting. The functor iRetry should return true if, after a timeout,
     the code should continue to wait.
     */
    template <typename F, typename FRETRY>
    void setupWorkerWithRetry(F&& iF, FRETRY&& iRetry) {
      using namespace boost::interprocess;
      scoped_lock<named_mutex> lock(mutex_);
      iF();
      using namespace boost::posix_time;
      //std::cout << id_ << " waiting for external process" << std::endl;
      bool shouldContinue = true;
      long long int retryCount = 0;
      do {
        if (not wait(lock)) {
          if (not iRetry()) {
            *stop_ = true;
            throw edm::Exception(edm::errors::ExternalFailure)
                << "Failed waiting for external process while setting up the process. Timed out after "
                << maxWaitInSeconds_ << " seconds with " << retryCount << " retries.";
          }
          //std::cerr<<"retrying\n";
          ++retryCount;
        } else {
          shouldContinue = false;
        }
      } while (shouldContinue);
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

    template <typename F, typename FRETRY>
    bool doTransitionWithRetry(F&& iF, FRETRY&& iRetry, edm::Transition iTrans, unsigned long long iTransitionID) {
      using namespace boost::interprocess;

      //std::cout << id_ << " taking from lock" << std::endl;
      scoped_lock<named_mutex> lock(mutex_);
      if (not wait(lock, iTrans, iTransitionID)) {
        if (not iRetry()) {
          return false;
        }
        bool shouldContinue = true;
        do {
          using namespace boost::posix_time;
          if (not continueWait(lock)) {
            if (not iRetry()) {
              return false;
            }
          } else {
            shouldContinue = false;
          }
        } while (shouldContinue);
      }
      //std::cout <<id_<<"running doTranstion command"<<std::endl;
      iF();
      return true;
    }

    ///This can be used with WriteBuffer to keep Controller and Worker in sync
    BufferInfo* toWorkerBufferInfo() { return toWorkerBufferInfo_; }
    ///This can be used with ReadBuffer to keep Controller and Worker in sync
    BufferInfo* fromWorkerBufferInfo() { return fromWorkerBufferInfo_; }

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

    unsigned int maxWaitInSeconds() const noexcept { return maxWaitInSeconds_; }

  private:
    struct CheckWorkerStatus {
      const unsigned long long initValue_;
      const unsigned long long* ptr_;

      [[nodiscard]] bool workerFinished() const noexcept { return initValue_ != *ptr_; }
    };

    [[nodiscard]] CheckWorkerStatus initCheckWorkerStatus(unsigned long long* iPtr) const noexcept {
      return {*iPtr, iPtr};
    }

    static BufferInfo* bufferInfo(const char* iWhich, boost::interprocess::managed_shared_memory& mem);

    std::string uniqueName(std::string iBase) const;

    bool wait(boost::interprocess::scoped_lock<boost::interprocess::named_mutex>& lock,
              edm::Transition iTrans,
              unsigned long long iTransID);
    bool wait(boost::interprocess::scoped_lock<boost::interprocess::named_mutex>& lock);
    bool continueWait(boost::interprocess::scoped_lock<boost::interprocess::named_mutex>& lock);

    // ---------- member data --------------------------------
    int id_;
    unsigned int maxWaitInSeconds_;
    std::string smName_;
    struct SMORemover {
      //handle removing the shared memory object from the system even
      // if an exception happens during construction
      SMORemover(const std::string& iName) : m_name(iName) {
        //remove an object which was left from a previous failed job
        boost::interprocess::shared_memory_object::remove(m_name.c_str());
      }
      ~SMORemover() { boost::interprocess::shared_memory_object::remove(m_name.c_str()); };
      //ControllerChannel passes in smName_ so it owns the string
      std::string const& m_name;
    } smRemover_;
    boost::interprocess::managed_shared_memory managed_sm_;
    BufferInfo* toWorkerBufferInfo_;
    BufferInfo* fromWorkerBufferInfo_;

    struct MutexRemover {
      MutexRemover(std::string iName) : m_name(std::move(iName)) {
        boost::interprocess::named_mutex::remove(m_name.c_str());
      }
      ~MutexRemover() { boost::interprocess::named_mutex::remove(m_name.c_str()); };
      std::string const m_name;
    };
    MutexRemover mutexRemover_;
    boost::interprocess::named_mutex mutex_;

    struct ConditionRemover {
      ConditionRemover(std::string iName) : m_name(std::move(iName)) {
        boost::interprocess::named_condition::remove(m_name.c_str());
      }
      ~ConditionRemover() { boost::interprocess::named_condition::remove(m_name.c_str()); };
      std::string const m_name;
    };

    ConditionRemover cndFromMainRemover_;
    boost::interprocess::named_condition cndFromMain_;

    ConditionRemover cndToMainRemover_;
    boost::interprocess::named_condition cndToMain_;

    edm::Transition* transitionType_;
    unsigned long long* transitionID_;
    bool* stop_;
    bool* keepEvent_;
  };
}  // namespace edm::shared_memory

#endif
