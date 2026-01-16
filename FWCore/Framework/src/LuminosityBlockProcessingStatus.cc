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
    eventSetupImpl_.reset();
    resumeGlobalLumiQueue();
  }

  void LuminosityBlockProcessingStatus::setGlobalEndRunHolder(WaitingTaskHolder holder) {
    globalEndRunHolder_ = std::move(holder);
  }

  bool LuminosityBlockProcessingStatus::shouldStreamStartLumi() {
    if (state_ == State::kNoMoreEvents)
      return false;

    bool changed = false;
    do {
      auto expected = State::kRunning;
      changed = state_.compare_exchange_strong(expected, State::kUpdating);
      if (expected == State::kNoMoreEvents)
        return false;
    } while (changed == false);

    ++nStreamsProcessingLumi_;
    state_ = State::kRunning;
    return true;
  }

  void LuminosityBlockProcessingStatus::noMoreEventsInLumi() {
    bool changed = false;
    do {
      auto expected = State::kRunning;
      changed = state_.compare_exchange_strong(expected, State::kUpdating);
      assert(expected != State::kNoMoreEvents);
    } while (changed == false);
    nStreamsStillProcessingLumi_.store(nStreamsProcessingLumi_);
    state_ = State::kNoMoreEvents;
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
