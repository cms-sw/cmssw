// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     LuminosityBlockProcessingStatus
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 11 Jan 2018 16:41:46 GMT
//

#include "LuminosityBlockProcessingStatus.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

namespace edm {
  void LuminosityBlockProcessingStatus::resetResources() {
    endIOVWaitingTasks_.doneWaiting(std::exception_ptr{});
    for (auto& iter : eventSetupImpls_) {
      iter.reset();
    }
    resumeGlobalLumiQueue();
  }

  void LuminosityBlockProcessingStatus::setGlobalEndRunHolder(WaitingTaskHolder holder) {
    globalEndRunHolder_ = std::move(holder);
  }

  void LuminosityBlockProcessingStatus::setEndTime() {
    constexpr char kUnset = 0;
    constexpr char kSetting = 1;
    constexpr char kSet = 2;

    if (endTimeSetStatus_ != kSet) {
      //not already set
      char expected = kUnset;
      if (endTimeSetStatus_.compare_exchange_strong(expected, kSetting)) {
        lumiPrincipal_->setEndTime(endTime_);
        endTimeSetStatus_.store(kSet);
      } else {
        //wait until time is set
        while (endTimeSetStatus_.load() != kSet) {
        }
      }
    }
  }
}  // namespace edm
