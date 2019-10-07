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

// system include files

// user include files
#include "LuminosityBlockProcessingStatus.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

namespace edm {
  void LuminosityBlockProcessingStatus::resetResources() {
    endIOVWaitingTasks_.doneWaiting(std::exception_ptr{});
    for (auto& iter : eventSetupImpls_) {
      iter.reset();
    }
    resumeGlobalLumiQueue();
    run_.reset();
  }

  void LuminosityBlockProcessingStatus::setEndTime() {
    if (2 != endTimeSetStatus_) {
      //not already set
      char expected = 0;
      if (endTimeSetStatus_.compare_exchange_strong(expected, 1)) {
        lumiPrincipal_->setEndTime(endTime_);
        endTimeSetStatus_.store(2);
      } else {
        //wait until time is set
        while (2 != endTimeSetStatus_.load()) {
        }
      }
    }
  }
}  // namespace edm
