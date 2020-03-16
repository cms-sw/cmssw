#ifndef FWCore_Framework_globalTransitionAsync_h
#define FWCore_Framework_globalTransitionAsync_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Function:    globalTransitionAsync
//
/**\function globalTransitionAsync globalTransitionAsync.h "globalTransitionAsync.h"

 Description: Helper functions for handling asynchronous global transitions

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 06 Sep 2016 16:04:26 GMT
//

// system include files
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <memory>
#include <vector>

// user include files

// forward declarations

namespace edm {
  class LuminosityBlockPrincipal;
  class ProcessBlockPrincipal;
  class RunPrincipal;

  //This is code in common between beginStreamRun and beginGlobalLuminosityBlock
  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      LuminosityBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    iSubProcess.doBeginLuminosityBlockAsync(std::move(iHolder), iPrincipal, iTS, iEventSetupImpls);
  }

  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      RunPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    iSubProcess.doBeginRunAsync(std::move(iHolder), iPrincipal, iTS, iEventSetupImpls);
  }

  template <typename Traits>
  inline void subProcessDoGlobalBeginTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      ProcessBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    iSubProcess.doBeginProcessBlockAsync<Traits>(std::move(iHolder), iPrincipal, iTS, iEventSetupImpls);
  }

  inline void subProcessDoGlobalEndTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      LuminosityBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    iSubProcess.doEndLuminosityBlockAsync(
        std::move(iHolder), iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
  }

  inline void subProcessDoGlobalEndTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      RunPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    iSubProcess.doEndRunAsync(std::move(iHolder), iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
  }

  inline void subProcessDoGlobalEndTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      ProcessBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    iSubProcess.doEndProcessBlockAsync(std::move(iHolder), iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
  }

  template <typename Traits, typename P, typename SC>
  void beginGlobalTransitionAsync(
      WaitingTaskHolder iWait,
      Schedule& iSchedule,
      P& iPrincipal,
      IOVSyncValue const& iTS,
      EventSetupImpl const& iES,
      std::vector<std::shared_ptr<const EventSetupImpl>> const*
          iEventSetupImpls,  // always null for runs until we enable concurrent run processing
      ServiceToken const& token,
      SC& iSubProcesses) {
    //When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs = make_waiting_task(
        tbb::task::allocate_root(),
        [&iSubProcesses, iWait, &iPrincipal, iTS, iEventSetupImpls](std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            auto excpt = *iPtr;
            auto delayError =
                make_waiting_task(tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
            WaitingTaskHolder h(delayError);
            for (auto& subProcess : iSubProcesses) {
              subProcessDoGlobalBeginTransitionAsync<Traits>(h, subProcess, iPrincipal, iTS, iEventSetupImpls);
            }
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoGlobalBeginTransitionAsync<Traits>(iWait, subProcess, iPrincipal, iTS, iEventSetupImpls);
            }
          }
        });

    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h), iPrincipal, iES, token);
  }

  // The only purpose of this function is to create dummy values for the
  // EventSetup arguments needed by the generic function beginGlobalTransitionsAsync
  template <typename Traits, typename P, typename SC>
  void beginGlobalTransitionAsync(
      WaitingTaskHolder iWait, Schedule& iSchedule, P& iPrincipal, ServiceToken const& token, SC& iSubProcesses) {
    // There is no EventSetup for ProcessBlock transitions
    // so we just pass in dummy values that are not used.
    IOVSyncValue dummyIOVSyncValue;
    EventSetupImpl dummyEventSetupImpl;
    std::vector<std::shared_ptr<const EventSetupImpl>> const* dummyEventSetupImpls = nullptr;

    beginGlobalTransitionAsync<Traits, P, SC>(std::move(iWait),
                                              iSchedule,
                                              iPrincipal,
                                              dummyIOVSyncValue,
                                              dummyEventSetupImpl,
                                              dummyEventSetupImpls,
                                              token,
                                              iSubProcesses);
  }

  template <typename Traits, typename P, typename SC>
  void endGlobalTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                P& iPrincipal,
                                IOVSyncValue const& iTS,
                                EventSetupImpl const& iES,
                                std::vector<std::shared_ptr<const EventSetupImpl>> const*
                                    iEventSetupImpls,  // always null for runs until we enable concurrent run processing
                                ServiceToken const& token,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException) {
    //When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs =
        make_waiting_task(tbb::task::allocate_root(),
                          [&iSubProcesses, iWait, &iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException](
                              std::exception_ptr const* iPtr) mutable {
                            if (iPtr) {
                              auto excpt = *iPtr;
                              auto delayError = make_waiting_task(
                                  tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
                              WaitingTaskHolder h(delayError);
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalEndTransitionAsync(
                                    h, subProcess, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
                              }
                            } else {
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalEndTransitionAsync(
                                    iWait, subProcess, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
                              }
                            }
                          });

    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h), iPrincipal, iES, token, cleaningUpAfterException);
  }

  // The only purpose of this function is to create dummy values for the
  // EventSetup arguments needed by the generic function endGlobalTransitionsAsync
  template <typename Traits, typename P, typename SC>
  void endGlobalTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                P& iPrincipal,
                                ServiceToken const& token,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException) {
    // There is no EventSetup for ProcessBlock transitions
    // so we just pass in dummy values that are not used.
    IOVSyncValue dummyIOVSyncValue;
    EventSetupImpl dummyEventSetupImpl;
    std::vector<std::shared_ptr<const EventSetupImpl>> const* dummyEventSetupImpls = nullptr;

    endGlobalTransitionAsync<Traits, P, SC>(std::move(iWait),
                                            iSchedule,
                                            iPrincipal,
                                            dummyIOVSyncValue,
                                            dummyEventSetupImpl,
                                            dummyEventSetupImpls,
                                            token,
                                            iSubProcesses,
                                            cleaningUpAfterException);
  }
};  // namespace edm

#endif
