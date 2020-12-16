#ifndef FWCore_Framework_streamTransitionAsync_h
#define FWCore_Framework_streamTransitionAsync_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Function:    streamTransitionAsync
//
/**\function streamTransitionAsync streamTransitionAsync.h "streamTransitionAsync.h"

 Description: Helper functions for handling asynchronous stream transitions

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

#include <vector>

namespace edm {

  //This is code in common between beginStreamRun and beginStreamLuminosityBlock
  inline void subProcessDoStreamBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     unsigned int i,
                                                     LumiTransitionInfo const& iTransitionInfo) {
    iSubProcess.doStreamBeginLuminosityBlockAsync(std::move(iHolder), i, iTransitionInfo);
  }

  inline void subProcessDoStreamBeginTransitionAsync(WaitingTaskHolder iHolder,
                                                     SubProcess& iSubProcess,
                                                     unsigned int i,
                                                     RunTransitionInfo const& iTransitionInfo) {
    iSubProcess.doStreamBeginRunAsync(std::move(iHolder), i, iTransitionInfo);
  }

  inline void subProcessDoStreamEndTransitionAsync(WaitingTaskHolder iHolder,
                                                   SubProcess& iSubProcess,
                                                   unsigned int i,
                                                   LumiTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    iSubProcess.doStreamEndLuminosityBlockAsync(std::move(iHolder), i, iTransitionInfo, cleaningUpAfterException);
  }

  inline void subProcessDoStreamEndTransitionAsync(WaitingTaskHolder iHolder,
                                                   SubProcess& iSubProcess,
                                                   unsigned int i,
                                                   RunTransitionInfo const& iTransitionInfo,
                                                   bool cleaningUpAfterException) {
    iSubProcess.doStreamEndRunAsync(std::move(iHolder), i, iTransitionInfo, cleaningUpAfterException);
  }

  template <typename Traits>
  void beginStreamTransitionAsync(WaitingTaskHolder iWait,
                                  Schedule& iSchedule,
                                  unsigned int iStreamIndex,
                                  typename Traits::TransitionInfoType& transitionInfo,
                                  ServiceToken const& token,
                                  std::vector<SubProcess>& iSubProcesses) {
    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    //NOTE: The subprocesses set their own service tokens
    auto subs = make_waiting_task(
        tbb::task::allocate_root(),
        [&iSubProcesses, iWait, iStreamIndex, info = transitionInfo](std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            auto excpt = *iPtr;
            auto delayError =
                make_waiting_task(tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
            WaitingTaskHolder h(delayError);
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamBeginTransitionAsync(h, subProcess, iStreamIndex, info);
            };
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamBeginTransitionAsync(iWait, subProcess, iStreamIndex, info);
            };
          }
        });

    WaitingTaskHolder h(subs);
    iSchedule.processOneStreamAsync<Traits>(std::move(h), iStreamIndex, transitionInfo, token);
  }

  template <typename Traits>
  void beginStreamsTransitionAsync(WaitingTaskHolder iWait,
                                   Schedule& iSchedule,
                                   unsigned int iNStreams,
                                   typename Traits::TransitionInfoType& transitionInfo,
                                   ServiceToken const& token,
                                   std::vector<SubProcess>& iSubProcesses) {
    for (unsigned int i = 0; i < iNStreams; ++i) {
      beginStreamTransitionAsync<Traits>(iWait, iSchedule, i, transitionInfo, token, iSubProcesses);
    }
  }

  template <typename Traits>
  void endStreamTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                unsigned int iStreamIndex,
                                typename Traits::TransitionInfoType& transitionInfo,
                                ServiceToken const& token,
                                std::vector<SubProcess>& iSubProcesses,
                                bool cleaningUpAfterException) {
    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    //NOTE: The subprocesses set their own service tokens

    auto subs = make_waiting_task(
        tbb::task::allocate_root(),
        [&iSubProcesses, iWait, iStreamIndex, info = transitionInfo, cleaningUpAfterException](
            std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            auto excpt = *iPtr;
            auto delayError =
                make_waiting_task(tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
            WaitingTaskHolder h(delayError);
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamEndTransitionAsync(h, subProcess, iStreamIndex, info, cleaningUpAfterException);
            }
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamEndTransitionAsync(iWait, subProcess, iStreamIndex, info, cleaningUpAfterException);
            }
          }
        });

    iSchedule.processOneStreamAsync<Traits>(
        WaitingTaskHolder(subs), iStreamIndex, transitionInfo, token, cleaningUpAfterException);
  }

  template <typename Traits>
  void endStreamsTransitionAsync(WaitingTaskHolder iWait,
                                 Schedule& iSchedule,
                                 unsigned int iNStreams,
                                 typename Traits::TransitionInfoType& transitionInfo,
                                 ServiceToken const& iToken,
                                 std::vector<SubProcess>& iSubProcesses,
                                 bool cleaningUpAfterException) {
    for (unsigned int i = 0; i < iNStreams; ++i) {
      endStreamTransitionAsync<Traits>(
          iWait, iSchedule, i, transitionInfo, iToken, iSubProcesses, cleaningUpAfterException);
    }
  }
};  // namespace edm

#endif
