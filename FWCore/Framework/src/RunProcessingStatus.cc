// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     RunProcessingStatus
//
// Original Author:  W. David Dagenhart
//         Created:  7 October 2021

#include "RunProcessingStatus.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

namespace edm {
  RunProcessingStatus::RunProcessingStatus(unsigned int iNStreams, WaitingTaskHolder const& holder)
      : holderOfTaskInProcessRuns_(holder),
        nStreamsStillProcessingBeginRun_(iNStreams),
        nStreamsStillProcessingRun_(iNStreams) {}

  std::optional<WaitingTaskHolder> RunProcessingStatus::releaseHolderOfTaskInProcessRuns() {
    if (not holderOfTaskInProcessRunsIsDone_.exchange(true)) {
      return std::move(holderOfTaskInProcessRuns_);
    }
    return std::nullopt;
  }
  void RunProcessingStatus::setHolderOfTaskInProcessRuns(WaitingTaskHolder const& holder) {
    holderOfTaskInProcessRuns_ = holder;
    holderOfTaskInProcessRunsIsDone_ = false;
  }
  void RunProcessingStatus::setHolderOfTaskInProcessRunsDoneWaiting() {
    auto temp = releaseHolderOfTaskInProcessRuns();
    if (temp) {
      temp->doneWaiting(std::exception_ptr{});
    }
  }

  void RunProcessingStatus::resetBeginResources() {
    endIOVWaitingTasks_.doneWaiting(std::exception_ptr{});
    eventSetupImpl_.reset();
  }

  void RunProcessingStatus::resetEndResources() {
    endIOVWaitingTasksEndRun_.doneWaiting(std::exception_ptr{});
    eventSetupImplEndRun_.reset();
  }

  void RunProcessingStatus::setEndTime() {
    constexpr char kUnset = 0;
    constexpr char kSetting = 1;
    constexpr char kSet = 2;

    if (endTimeSetStatus_ != kSet) {
      //not already set
      char expected = kUnset;
      if (endTimeSetStatus_.compare_exchange_strong(expected, kSetting)) {
        runPrincipal_->setEndTime(endTime_);
        endTimeSetStatus_.store(kSet);
      } else {
        //wait until time is set
        while (endTimeSetStatus_.load() != kSet) {
        }
      }
    }
  }
}  // namespace edm
