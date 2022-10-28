// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
//
// Implementation:
//     [Notes on implementation]
//
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Framework/interface/global/OutputModuleBase.h"

#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"

namespace edm {
  namespace global {

    // -------------------------------------------------------
    OutputModuleBase::OutputModuleBase(ParameterSet const& pset) : core::OutputModuleCore(pset) {}

    void OutputModuleBase::doPreallocate(PreallocationConfiguration const& iPC) {
      auto nstreams = iPC.numberOfStreams();
      preallocStreams(nstreams);
      core::OutputModuleCore::doPreallocate_(iPC);
      preallocate(iPC);
    }

    void OutputModuleBase::doBeginJob() { core::OutputModuleCore::doBeginJob_(); }

    bool OutputModuleBase::doEvent(EventTransitionInfo const& info,
                                   ActivityRegistry* act,
                                   ModuleCallingContext const* mcc) {
      { core::OutputModuleCore::doEvent_(info, act, mcc); }

      auto remainingEvents = remainingEvents_.load();
      bool keepTrying = remainingEvents > 0;
      while (keepTrying) {
        auto newValue = remainingEvents - 1;
        keepTrying = !remainingEvents_.compare_exchange_strong(remainingEvents, newValue);
        if (keepTrying) {
          // the exchange failed because the value was changed by another thread.
          // remainingEvents was changed to be the new value of remainingEvents_;
          keepTrying = remainingEvents > 0;
        }
      }
      return true;
    }

    void OutputModuleBase::doAcquire(EventTransitionInfo const& info,
                                     ActivityRegistry* act,
                                     ModuleCallingContext const* mcc,
                                     WaitingTaskWithArenaHolder& holder) {
      EventForOutput e(info, moduleDescription(), mcc);
      e.setConsumer(this);
      EventAcquireSignalsSentry sentry(act, mcc);
      this->doAcquire_(e.streamID(), e, holder);
    }
  }  // namespace global
}  // namespace edm
