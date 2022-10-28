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

#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/chain_first.h"

#include <exception>
#include <utility>
#include <vector>

namespace edm {

  //This is code in common between beginStreamRun and beginGlobalLuminosityBlock
  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     LumiTransitionInfo const& iTransitionInfo,
                                                     bool) {
    iSubProcess.doBeginLuminosityBlockAsync(std::move(iHolder), iTransitionInfo);
  }

  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     RunTransitionInfo const& iTransitionInfo,
                                                     bool) {
    iSubProcess.doBeginRunAsync(std::move(iHolder), iTransitionInfo);
  }

  template <typename Traits>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     ProcessBlockTransitionInfo const& iTransitionInfo,
                                                     bool cleaningUpAfterException) {
    iSubProcess.doBeginProcessBlockAsync<Traits>(std::move(iHolder), iTransitionInfo, cleaningUpAfterException);
  }

  inline void subProcessDoGlobalEndTransitionAsync(WaitingTaskHolder iHolder,
                                                   SubProcess& iSubProcess,
                                                   LumiTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    iSubProcess.doEndLuminosityBlockAsync(std::move(iHolder), iTransitionInfo, cleaningUpAfterException);
  }

  inline void subProcessDoGlobalEndTransitionAsync(WaitingTaskHolder iHolder,
                                                   SubProcess& iSubProcess,
                                                   RunTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    iSubProcess.doEndRunAsync(std::move(iHolder), iTransitionInfo, cleaningUpAfterException);
  }

  inline void subProcessDoGlobalEndTransitionAsync(WaitingTaskHolder iHolder,
                                                   SubProcess& iSubProcess,
                                                   ProcessBlockTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    iSubProcess.doEndProcessBlockAsync(std::move(iHolder), iTransitionInfo, cleaningUpAfterException);
  }

  template <typename Traits>
  void beginGlobalTransitionAsync(WaitingTaskHolder iWait,
                                  Schedule& iSchedule,
                                  typename Traits::TransitionInfoType& transitionInfo,
                                  ServiceToken const& token,
                                  std::vector<SubProcess>& iSubProcesses,
                                  bool cleaningUpAfterException = false) {
    // When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    using namespace edm::waiting_task;

    chain::first([&](auto nextTask) {
      iSchedule.processOneGlobalAsync<Traits>(std::move(nextTask), transitionInfo, token, cleaningUpAfterException);
    }) |
        chain::then([&iSubProcesses, info = transitionInfo, cleaningUpAfterException](std::exception_ptr const* iPtr,
                                                                                      auto nextTask) {
          if (iPtr) {
            //delay handling exception until after subProcesses run
            chain::first([&](auto nextTask) {
              for (auto& subProcess : iSubProcesses) {
                subProcessDoGlobalBeginTransitionAsync<Traits>(nextTask, subProcess, info, cleaningUpAfterException);
              }
            }) | chain::then([excpt = *iPtr](std::exception_ptr const*, auto nextTask) {
              nextTask.doneWaiting(excpt);
            }) | chain::runLast(nextTask);
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoGlobalBeginTransitionAsync<Traits>(nextTask, subProcess, info, cleaningUpAfterException);
            }
          }
        }) |
        chain::runLast(iWait);
  }

  template <typename Traits>
  void endGlobalTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                typename Traits::TransitionInfoType& transitionInfo,
                                ServiceToken const& token,
                                std::vector<SubProcess>& iSubProcesses,
                                bool cleaningUpAfterException) {
    using namespace edm::waiting_task;
    chain::first([&](auto nextTask) {
      iSchedule.processOneGlobalAsync<Traits>(std::move(nextTask), transitionInfo, token, cleaningUpAfterException);
    })
        // When we are done processing the global for this process,
        // we need to run the global for all SubProcesses
        | chain::then([&iSubProcesses, info = transitionInfo, cleaningUpAfterException](std::exception_ptr const* iPtr,
                                                                                        auto nextTask) {
            if (iPtr) {
              //still run the sub process but pass this exception to the nextTask
              auto excpt = *iPtr;
              chain::first([&](auto nextTask) {
                for (auto& subProcess : iSubProcesses) {
                  subProcessDoGlobalEndTransitionAsync(nextTask, subProcess, info, cleaningUpAfterException);
                }
              }) | chain::then([excpt](std::exception_ptr const*, auto nextTask) { nextTask.doneWaiting(excpt); }) |
                  chain::runLast(std::move(nextTask));
            } else {
              for (auto& subProcess : iSubProcesses) {
                subProcessDoGlobalEndTransitionAsync(nextTask, subProcess, info, cleaningUpAfterException);
              }
            }
          }) |
        chain::runLast(iWait);
  }

};  // namespace edm

#endif
