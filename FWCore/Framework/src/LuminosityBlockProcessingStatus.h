#ifndef FWCore_Framework_LuminosityBlockProcessingStatus_h
#define FWCore_Framework_LuminosityBlockProcessingStatus_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     LuminosityBlockProcessingStatus
//
/**\class edm::LuminosityBlockProcessingStatus

 Description: Keep status information about one LuminosityBlock transition

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 19 Dec 2017 14:24:57 GMT
//

// system include files
#include <memory>
#include <atomic>
#include <vector>

// user include files
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

// forward declarations
namespace edm {
#if !defined(TEST_NO_FWD_DECL)
  class LuminosityBlockPrincipal;
  class LuminosityBlockProcessingStatus;
#endif

  class LuminosityBlockProcessingStatus {
  public:
    LuminosityBlockProcessingStatus(unsigned int iNStreams) : nStreamsStillProcessingLumi_(iNStreams) {}

    LuminosityBlockProcessingStatus(LuminosityBlockProcessingStatus const&) = delete;
    LuminosityBlockProcessingStatus const& operator=(LuminosityBlockProcessingStatus const&) = delete;

    ~LuminosityBlockProcessingStatus() { endIOVWaitingTasks_.doneWaiting(std::exception_ptr{}); }

    void setResumer(LimitedTaskQueue::Resumer iResumer) { globalLumiQueueResumer_ = std::move(iResumer); }
    void resumeGlobalLumiQueue() {
      //free lumi for next usage
      lumiPrincipal_.reset();
      globalLumiQueueResumer_.resume();
    }

    void resetResources();

    std::shared_ptr<LuminosityBlockPrincipal>& lumiPrincipal() { return lumiPrincipal_; }
    void setLumiPrincipal(std::shared_ptr<LuminosityBlockPrincipal> val) { lumiPrincipal_ = std::move(val); }

    EventSetupImpl const& eventSetupImpl(unsigned subProcessIndex) const {
      return *eventSetupImpls_.at(subProcessIndex);
    }

    std::vector<std::shared_ptr<const EventSetupImpl>>& eventSetupImpls() { return eventSetupImpls_; }
    std::vector<std::shared_ptr<const EventSetupImpl>> const& eventSetupImpls() const { return eventSetupImpls_; }

    WaitingTaskList& endIOVWaitingTasks() { return endIOVWaitingTasks_; }

    void setGlobalEndRunHolder(WaitingTaskHolder);
    void globalEndRunHolderDoneWaiting() { globalEndRunHolder_.doneWaiting(std::exception_ptr{}); }

    bool streamFinishedLumi() { return 0 == (--nStreamsStillProcessingLumi_); }

    //These should only be called while in the InputSource's task queue
    void updateLastTimestamp(edm::Timestamp const& iTime) {
      if (iTime > endTime_) {
        endTime_ = iTime;
      }
    }
    edm::Timestamp const& lastTimestamp() const { return endTime_; }

    //Called once all events in Lumi have been processed
    void setEndTime();

    enum class EventProcessingState { kProcessing, kPauseForFileTransition, kStopLumi };
    EventProcessingState eventProcessingState() const { return eventProcessingState_; }
    void setEventProcessingState(EventProcessingState val) { eventProcessingState_ = val; }

    bool haveStartedNextLumiOrEndedRun() const { return startedNextLumiOrEndedRun_; }
    void startNextLumiOrEndRun() { startedNextLumiOrEndedRun_ = true; }

    bool didGlobalBeginSucceed() const { return globalBeginSucceeded_; }
    void globalBeginDidSucceed() { globalBeginSucceeded_ = true; }

    bool cleaningUpAfterException() const { return cleaningUpAfterException_; }
    void setCleaningUpAfterException(bool value) { cleaningUpAfterException_ = value; }

  private:
    // ---------- member data --------------------------------
    LimitedTaskQueue::Resumer globalLumiQueueResumer_;
    std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;
    std::vector<std::shared_ptr<const EventSetupImpl>> eventSetupImpls_;
    WaitingTaskList endIOVWaitingTasks_;
    edm::WaitingTaskHolder globalEndRunHolder_;
    std::atomic<unsigned int> nStreamsStillProcessingLumi_{0};  //read/write as streams finish lumi so must be atomic
    edm::Timestamp endTime_{};
    std::atomic<char> endTimeSetStatus_{0};
    EventProcessingState eventProcessingState_{EventProcessingState::kProcessing};
    bool startedNextLumiOrEndedRun_{false};  //read/write in m_sourceQueue
    bool globalBeginSucceeded_{false};
    bool cleaningUpAfterException_{false};
  };
}  // namespace edm

#endif
