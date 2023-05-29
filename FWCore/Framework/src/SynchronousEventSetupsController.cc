// -*- C++ -*-
//
// Package:     Framework
// Class  :     SynchronousEventSetupsController
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Wed Jan 12 14:30:44 CST 2011
//

#include "FWCore/Framework/src/SynchronousEventSetupsController.h"

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

#include <algorithm>
#include <iostream>
#include <set>

namespace edm {
  namespace eventsetup {

    SynchronousEventSetupsController::SynchronousEventSetupsController()
        : globalControl_(oneapi::tbb::global_control::max_allowed_parallelism, 1) {}

    SynchronousEventSetupsController::~SynchronousEventSetupsController() {
      FinalWaitingTask finalTask{taskGroup_};
      controller_.endIOVsAsync(edm::WaitingTaskHolder(taskGroup_, &finalTask));
      finalTask.waitNoThrow();
    }

    std::shared_ptr<EventSetupProvider> SynchronousEventSetupsController::makeProvider(
        ParameterSet& iPSet, ActivityRegistry* activityRegistry, ParameterSet const* eventSetupPset) {
      return controller_.makeProvider(iPSet, activityRegistry, eventSetupPset);
    }

    void SynchronousEventSetupsController::eventSetupForInstance(IOVSyncValue const& syncValue) {
      synchronousEventSetupForInstance(syncValue, taskGroup_, controller_);
    }

  }  // namespace eventsetup
}  // namespace edm
