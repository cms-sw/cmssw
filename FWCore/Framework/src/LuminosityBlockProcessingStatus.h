#ifndef FWCore_Framework_LuminosityBlockProcessingStatus_h
#define FWCore_Framework_LuminosityBlockProcessingStatus_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     LuminosityBlockProcessingStatus
//
/**\class LuminosityBlockProcessingStatus LuminosityBlockProcessingStatus.h "LuminosityBlockProcessingStatus.h"

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
  class EventProcessor;
  class LuminosityBlockPrincipal;
  class LuminosityBlockProcessingStatus;
#endif

  class LuminosityBlockProcessingStatus {
  public:
    LuminosityBlockProcessingStatus(EventProcessor* iEP, unsigned int iNStreams, std::shared_ptr<void> iRunResource)
        : run_(std::move(iRunResource)), eventProcessor_(iEP), nStreamsStillProcessingLumi_(iNStreams) {}

    LuminosityBlockProcessingStatus(LuminosityBlockProcessingStatus const&) = delete;
    LuminosityBlockProcessingStatus const& operator=(LuminosityBlockProcessingStatus const&) = delete;

    ~LuminosityBlockProcessingStatus() { endIOVWaitingTasks_.doneWaiting(std::exception_ptr{}); }

    std::shared_ptr<LuminosityBlockPrincipal>& lumiPrincipal() { return lumiPrincipal_; }

    void setResumer(LimitedTaskQueue::Resumer iResumer) { globalLumiQueueResumer_ = std::move(iResumer); }
    void resumeGlobalLumiQueue() {
      //free lumi for next usage
      lumiPrincipal_.reset();
      globalLumiQueueResumer_.resume();
    }

    void resetResources();

    EventSetupImpl const& eventSetupImpl(unsigned subProcessIndex) const {
      return *eventSetupImpls_.at(subProcessIndex);
    }

    std::vector<std::shared_ptr<const EventSetupImpl>>& eventSetupImpls() { return eventSetupImpls_; }
    std::vector<std::shared_ptr<const EventSetupImpl>> const& eventSetupImpls() const { return eventSetupImpls_; }

    WaitingTaskList& endIOVWaitingTasks() { return endIOVWaitingTasks_; }

    bool streamFinishedLumi() { return 0 == (--nStreamsStillProcessingLumi_); }

    bool wasEventProcessingStopped() const { return stopProcessingEvents_; }
    void stopProcessingEvents() { stopProcessingEvents_ = true; }
    void startProcessingEvents() { stopProcessingEvents_ = false; }

    bool isLumiEnding() const { return lumiEnding_; }
    void endLumi() { lumiEnding_ = true; }

    bool continuingLumi() const { return continuingLumi_; }
    void haveContinuedLumi() { continuingLumi_ = false; }
    void needToContinueLumi() { continuingLumi_ = true; }

    bool haveStartedNextLumi() const { return startedNextLumi_; }
    void startNextLumi() { startedNextLumi_ = true; }

    bool didGlobalBeginSucceed() const { return globalBeginSucceeded_; }
    void globalBeginDidSucceed() { globalBeginSucceeded_ = true; }

    void noExceptionHappened() { cleaningUpAfterException_ = false; }
    bool cleaningUpAfterException() const { return cleaningUpAfterException_; }

    //These should only be called while in the InputSource's task queue
    void updateLastTimestamp(edm::Timestamp const& iTime) {
      if (iTime > endTime_) {
        endTime_ = iTime;
      }
    }
    edm::Timestamp const& lastTimestamp() const { return endTime_; }

    void setNextSyncValue(IOVSyncValue iValue) { nextSyncValue_ = std::move(iValue); }

    const IOVSyncValue nextSyncValue() const { return nextSyncValue_; }

    std::shared_ptr<void> const& runResource() const { return run_; }

    //Called once all events in Lumi have been processed
    void setEndTime();

  private:
    // ---------- member data --------------------------------
    std::shared_ptr<void> run_;
    LimitedTaskQueue::Resumer globalLumiQueueResumer_;
    std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;
    std::vector<std::shared_ptr<const EventSetupImpl>> eventSetupImpls_;
    WaitingTaskList endIOVWaitingTasks_;
    EventProcessor* eventProcessor_ = nullptr;
    IOVSyncValue nextSyncValue_;
    std::atomic<unsigned int> nStreamsStillProcessingLumi_{0};  //read/write as streams finish lumi so must be atomic
    edm::Timestamp endTime_{};
    std::atomic<char> endTimeSetStatus_{0};
    bool stopProcessingEvents_{false};  //read/write in m_sourceQueue OR from main thread when no tasks running
    bool lumiEnding_{
        false};  //read/write in m_sourceQueue NOTE: This is a useful cache instead of recalculating each call
    bool continuingLumi_{false};   //read/write in m_sourceQueue OR from main thread when no tasks running
    bool startedNextLumi_{false};  //read/write in m_sourceQueue
    bool globalBeginSucceeded_{false};
    bool cleaningUpAfterException_{true};
  };
}  // namespace edm

#endif
