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
#include "FWCore/Framework/src/SubProcess.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <exception>
#include <utility>
#include <vector>

namespace edm {

  //This is code in common between beginStreamRun and beginGlobalLuminosityBlock
  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     LumiTransitionInfo const& iTransitionInfo) {
    iSubProcess.doBeginLuminosityBlockAsync(std::move(iHolder), iTransitionInfo);
  }

  template <typename T>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     RunTransitionInfo const& iTransitionInfo) {
    iSubProcess.doBeginRunAsync(std::move(iHolder), iTransitionInfo);
  }

  template <typename Traits>
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     ProcessBlockTransitionInfo const& iTransitionInfo) {
    iSubProcess.doBeginProcessBlockAsync<Traits>(std::move(iHolder), iTransitionInfo);
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
                                  std::vector<SubProcess>& iSubProcesses) {
    // When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs =
        make_waiting_task(tbb::task::allocate_root(),
                          [&iSubProcesses, iWait, info = transitionInfo](std::exception_ptr const* iPtr) mutable {
                            if (iPtr) {
                              auto excpt = *iPtr;
                              auto delayError = make_waiting_task(
                                  tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
                              WaitingTaskHolder h(delayError);
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalBeginTransitionAsync<Traits>(h, subProcess, info);
                              }
                            } else {
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalBeginTransitionAsync<Traits>(iWait, subProcess, info);
                              }
                            }
                          });

    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h), transitionInfo, token);
  }

  template <typename Traits>
  void endGlobalTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                typename Traits::TransitionInfoType& transitionInfo,
                                ServiceToken const& token,
                                std::vector<SubProcess>& iSubProcesses,
                                bool cleaningUpAfterException) {
    // When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs =
        make_waiting_task(tbb::task::allocate_root(),
                          [&iSubProcesses, iWait, info = transitionInfo, cleaningUpAfterException](
                              std::exception_ptr const* iPtr) mutable {
                            if (iPtr) {
                              auto excpt = *iPtr;
                              auto delayError = make_waiting_task(
                                  tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
                              WaitingTaskHolder h(delayError);
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalEndTransitionAsync(h, subProcess, info, cleaningUpAfterException);
                              }
                            } else {
                              for (auto& subProcess : iSubProcesses) {
                                subProcessDoGlobalEndTransitionAsync(iWait, subProcess, info, cleaningUpAfterException);
                              }
                            }
                          });

    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h), transitionInfo, token, cleaningUpAfterException);
  }

};  // namespace edm

#endif
