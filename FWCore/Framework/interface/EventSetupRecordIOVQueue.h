// -*- C++ -*-
#ifndef FWCore_Framework_EventSetupRecordIOVQueue_h
#define FWCore_Framework_EventSetupRecordIOVQueue_h
//
// Package:     Framework
// Class  :     EventSetupRecordIOVQueue
//
/** \class edm::eventsetup::EventSetupRecordIOVQueue

 Description: Contains a LimitedQueue for each Record that
 limits the number of IOVs that can be processed concurrently.
 Also contains data closely related to starting and
 stopping those IOVs.

 This manages creating the TBB tasks that start and end IOVs.
 It creates those tasks. It ensures those tasks wait until
 the proper time to start and notify the proper other tasks
 when done.

 It protects the many EventSetupRecordImpl's from data races
 by ensuring that while multiple threads are using them
 concurrently only thread safe functions are being called.

*/
//
// Original Author: W. David Dagenhart
//          Created: 22 Feb 2019
//
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"

#include <atomic>
#include <memory>
#include <vector>
#include <optional>

namespace edm {
  namespace eventsetup {

    class EventSetupRecordProvider;

    class EventSetupRecordIOVQueue {
    public:
      enum class IntervalStatus { NotInitializedForSyncValue, Invalid, NewInterval, UpdateIntervalEnd, SameInterval };

      EventSetupRecordIOVQueue(unsigned int nConcurrentIOVs);
      ~EventSetupRecordIOVQueue();

      void endIOVAsync(WaitingTaskHolder endTask);

      ///returns true if the IOV is different than previously, including if it changed from valid to invalid.
      bool setValidityIntervalFor(const IOVSyncValue& iTime);

      void checkForNewIOVsAndStartIfNeededAsync(WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                WaitingTaskList& endIOVWaitingTasks,
                                                bool newEventSetupImpl,
                                                EventSetupImpl& eventSetupImpl);

      void startNewIOVAsync(WaitingTaskHolder const& taskToStartAfterIOVInit,
                            WaitingTaskList& endIOVWaitingTasks,
                            EventSetupImpl& eventSetupImpl);

      void addRecProvider(EventSetupRecordProvider* recProvider);

      auto const& key() const { return recordProvider_->key(); }
      void reset();

    private:
      void updateValidityIntervalAndStatus(const IOVSyncValue& iTime);

      // Used to limit the number of concurrent IOVs
      edm::LimitedTaskQueue iovQueue_;
      edm::WaitingTaskList endIOVTasks_;

      // Each element of this vector corresponds to one IOV
      // out the set of possible concurrent IOVs.
      std::vector<std::atomic<bool>> isAvailable_;

      edm::IOVSyncValue firstForCurrentIOV_ = edm::IOVSyncValue::invalidIOVSyncValue();
      edm::propagate_const<EventSetupRecordProvider*> recordProvider_;

      // These are associated with the most recent iov.
      unsigned long long cacheIdentifier_;
      WaitingTaskHolder endIOVTaskHolder_;
      // This holds the interval most recently initialized with a call to
      // eventSetupForInstance. A particular EventSetupRecordImpl in flight
      // might contain an older interval.
      ValidityInterval validityInterval_;
      IntervalStatus intervalStatus_ = IntervalStatus::Invalid;

      unsigned int lastUsedIovIndex_ = std::numeric_limits<unsigned int>::max();
      bool endIOVCalled_ = false;
      bool newIOVNeeded_ = false;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
