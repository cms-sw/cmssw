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
      : holderOfTaskInProcessRuns_(holder), nStreamsStillProcessingRun_(iNStreams) {}

  void RunProcessingStatus::resetBeginResources() {
    endIOVWaitingTasks_.doneWaiting(std::exception_ptr{});
    for (auto& iter : eventSetupImpls_) {
      iter.reset();
    }
  }

  void RunProcessingStatus::resetEndResources() {
    endIOVWaitingTasksEndRun_.doneWaiting(std::exception_ptr{});
    for (auto& iter : eventSetupImplsEndRun_) {
      iter.reset();
    }
  }

  void RunProcessingStatus::setEndTime() {
    if (2 != endTimeSetStatus_) {
      //not already set
      char expected = 0;
      if (endTimeSetStatus_.compare_exchange_strong(expected, 1)) {
        runPrincipal_->setEndTime(endTime_);
        endTimeSetStatus_.store(2);
      } else {
        //wait until time is set
        while (2 != endTimeSetStatus_.load()) {
        }
      }
    }
  }
}  // namespace edm
