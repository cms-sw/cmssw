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
#include <mutex>

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
    //there is no more work to do for this lumi and some other stream
    // is already processing this lumi
    if (haveStartedNextLumiOrEndedRun() and aStreamStartedLumi_) {
      return false;
    }
    {
      std::lock_guard g{checkShouldStartStream_};
      auto streamStartedLumi = aStreamStartedLumi_.load();
      if (not streamStartedLumi) {
        //at least one stream must process the lumi, and we are 'it'
        assert(0 == nStreamsProcessingLumi_);
        ++nStreamsProcessingLumi_;
        aStreamStartedLumi_.store(true);
        return true;
      }
      //check condition again as it could have changed since acquiring the lock
      if (haveStartedNextLumiOrEndedRun()) {
        return false;
      }
      //At this point, nStreamsProcessingLumi_ can't be zero since
      // - if this were the first lumi then aStreamStartedLumi_ would have been false
      //   and we wouldn't have gotten here
      // - if we made it to `streamFinishedLumi` and it decremented the value to 0
      //   that would happen while that routine held the spin lock and that function
      //   will not be called until `haveStartedNextLumiOrEndedRun()` is true
      //   therefore when this routine obtained the spin lock it would have passed
      //   the test above and not gotten here
      assert(0 != nStreamsProcessingLumi_);
      ++nStreamsProcessingLumi_;
    }
    return true;
  }

  bool LuminosityBlockProcessingStatus::streamFinishedLumi() {
    assert(haveStartedNextLumiOrEndedRun());
    std::lock_guard g{checkShouldStartStream_};
    //this must be done in the spin lock so that the changes to
    // aStreamStartedLumi_ and nStreamsProcessingLumi_ happen
    // atomically within shouldStreamStartLumi

    if (0 == --nStreamsProcessingLumi_) {
      return true;
    }
    return false;
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
