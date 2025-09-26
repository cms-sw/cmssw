// -*- C++ -*-
#ifndef FWCore_Framework_RunProcessingStatus_h
#define FWCore_Framework_RunProcessingStatus_h
//
// Package:     FWCore/Framework
// Class  :     RunProcessingStatus
//
/**\class edm::RunProcessingStatus

 Description: Keep status information about one Run transition

*/
//
// Original Author: W. David Dagenhart
//          Created: 1 Oct 2021
//

#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <atomic>
#include <memory>

namespace edm {

  class EventSetupImpl;
  class RunPrincipal;

  class RunProcessingStatus {
  public:
    RunProcessingStatus(unsigned int iNStreams, WaitingTaskHolder const& holder);

    RunProcessingStatus(RunProcessingStatus const&) = delete;
    RunProcessingStatus const& operator=(RunProcessingStatus const&) = delete;

    WaitingTaskHolder& holderOfTaskInProcessRuns() { return holderOfTaskInProcessRuns_; }
    void setHolderOfTaskInProcessRuns(WaitingTaskHolder const& holder) { holderOfTaskInProcessRuns_ = holder; }

    void setResumer(LimitedTaskQueue::Resumer iResumer) { globalRunQueueResumer_ = std::move(iResumer); }
    void resumeGlobalRunQueue() {
      //free run for next usage
      runPrincipal_.reset();
      globalRunQueueResumer_.resume();
    }

    std::shared_ptr<RunPrincipal>& runPrincipal() { return runPrincipal_; }
    void setRunPrincipal(std::shared_ptr<RunPrincipal> val) { runPrincipal_ = std::move(val); }

    void resetBeginResources();
    void resetEndResources();

    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

    std::shared_ptr<const EventSetupImpl>& eventSetupImplPtr() { return eventSetupImpl_; }

    EventSetupImpl const& eventSetupImplEndRun() const { return *eventSetupImplEndRun_; }

    std::shared_ptr<const EventSetupImpl>& eventSetupImplPtrEndRun() { return eventSetupImplEndRun_; }

    WaitingTaskList& endIOVWaitingTasks() { return endIOVWaitingTasks_; }

    WaitingTaskList& endIOVWaitingTasksEndRun() { return endIOVWaitingTasksEndRun_; }

    void setGlobalEndRunHolder(WaitingTaskHolder holder) { globalEndRunHolder_ = std::move(holder); }
    WaitingTaskHolder& globalEndRunHolder() { return globalEndRunHolder_; }

    bool streamFinishedBeginRun() { return 0 == (--nStreamsStillProcessingBeginRun_); }
    bool streamFinishedRun() { return 0 == (--nStreamsStillProcessingRun_); }

    //These should only be called while in the InputSource's task queue
    void updateLastTimestamp(edm::Timestamp const& iTime) {
      if (iTime > endTime_) {
        endTime_ = iTime;
      }
    }
    edm::Timestamp const& lastTimestamp() const { return endTime_; }

    void setEndTime();

    bool didGlobalBeginSucceed() const { return globalBeginSucceeded_; }
    void globalBeginDidSucceed() { globalBeginSucceeded_ = true; }

    bool cleaningUpAfterException() const { return cleaningUpAfterException_; }
    void setCleaningUpAfterException(bool val) { cleaningUpAfterException_ = val; }

    bool stopBeforeProcessingRun() const { return stopBeforeProcessingRun_; }
    void setStopBeforeProcessingRun(bool val) { stopBeforeProcessingRun_ = val; }

    bool endingEventSetupSucceeded() const { return endingEventSetupSucceeded_; }
    void setEndingEventSetupSucceeded(bool val) { endingEventSetupSucceeded_ = val; }

  private:
    WaitingTaskHolder holderOfTaskInProcessRuns_;
    LimitedTaskQueue::Resumer globalRunQueueResumer_;
    std::shared_ptr<RunPrincipal> runPrincipal_;
    std::shared_ptr<const EventSetupImpl> eventSetupImpl_;
    WaitingTaskList endIOVWaitingTasks_;
    std::shared_ptr<const EventSetupImpl> eventSetupImplEndRun_;
    WaitingTaskList endIOVWaitingTasksEndRun_;
    WaitingTaskHolder globalEndRunHolder_;
    std::atomic<unsigned int> nStreamsStillProcessingBeginRun_;
    std::atomic<unsigned int> nStreamsStillProcessingRun_;
    edm::Timestamp endTime_{};
    std::atomic<char> endTimeSetStatus_{0};
    bool globalBeginSucceeded_{false};
    bool cleaningUpAfterException_{false};
    bool stopBeforeProcessingRun_{false};
    bool endingEventSetupSucceeded_{true};
  };
}  // namespace edm

#endif
