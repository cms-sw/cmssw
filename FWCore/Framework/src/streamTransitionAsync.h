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

// system include files
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// user include files

// forward declarations

namespace edm {
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class RunPrincipal;

  //This is code in common between beginStreamRun and beginStreamLuminosityBlock
  inline void subProcessDoStreamBeginTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      unsigned int i,
      LuminosityBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    iSubProcess.doStreamBeginLuminosityBlockAsync(std::move(iHolder), i, iPrincipal, iTS, iEventSetupImpls);
  }

  inline void subProcessDoStreamBeginTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      unsigned int i,
      RunPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls) {
    iSubProcess.doStreamBeginRunAsync(std::move(iHolder), i, iPrincipal, iTS, iEventSetupImpls);
  }

  inline void subProcessDoStreamEndTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      unsigned int i,
      LuminosityBlockPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    iSubProcess.doStreamEndLuminosityBlockAsync(
        std::move(iHolder), i, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
  }

  inline void subProcessDoStreamEndTransitionAsync(
      WaitingTaskHolder iHolder,
      SubProcess& iSubProcess,
      unsigned int i,
      RunPrincipal& iPrincipal,
      IOVSyncValue const& iTS,
      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
      bool cleaningUpAfterException) {
    iSubProcess.doStreamEndRunAsync(std::move(iHolder), i, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
  }

  template <typename Traits, typename P, typename SC>
  void beginStreamTransitionAsync(
      WaitingTaskHolder iWait,
      Schedule& iSchedule,
      unsigned int iStreamIndex,
      P& iPrincipal,
      IOVSyncValue const& iTS,
      EventSetupImpl const& iES,
      std::vector<std::shared_ptr<const EventSetupImpl>> const*
          iEventSetupImpls,  // always null for runs until we enable concurrent run processing
      ServiceToken const& token,
      SC& iSubProcesses) {
    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    //NOTE: The subprocesses set their own service tokens
    auto subs = make_waiting_task(
        tbb::task::allocate_root(),
        [&iSubProcesses, iWait, iStreamIndex, &iPrincipal, iTS, iEventSetupImpls](
            std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            auto excpt = *iPtr;
            auto delayError =
                make_waiting_task(tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
            WaitingTaskHolder h(delayError);
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamBeginTransitionAsync(h, subProcess, iStreamIndex, iPrincipal, iTS, iEventSetupImpls);
            };
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamBeginTransitionAsync(
                  iWait, subProcess, iStreamIndex, iPrincipal, iTS, iEventSetupImpls);
            };
          }
        });

    WaitingTaskHolder h(subs);
    iSchedule.processOneStreamAsync<Traits>(std::move(h), iStreamIndex, iPrincipal, iES, token);
  }

  template <typename Traits, typename P, typename SC>
  void beginStreamsTransitionAsync(WaitingTask* iWait,
                                   Schedule& iSchedule,
                                   unsigned int iNStreams,
                                   P& iPrincipal,
                                   IOVSyncValue const& iTS,
                                   EventSetupImpl const& iES,
                                   std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
                                   ServiceToken const& token,
                                   SC& iSubProcesses) {
    WaitingTaskHolder holdUntilAllStreamsCalled(iWait);
    for (unsigned int i = 0; i < iNStreams; ++i) {
      beginStreamTransitionAsync<Traits>(
          WaitingTaskHolder(iWait), iSchedule, i, iPrincipal, iTS, iES, iEventSetupImpls, token, iSubProcesses);
    }
  }

  template <typename Traits, typename P, typename SC>
  void endStreamTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                unsigned int iStreamIndex,
                                P& iPrincipal,
                                IOVSyncValue const& iTS,
                                EventSetupImpl const& iES,
                                std::vector<std::shared_ptr<const EventSetupImpl>> const*
                                    iEventSetupImpls,  // always null for runs until we enable concurrent run processing
                                ServiceToken const& token,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException) {
    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    //NOTE: The subprocesses set their own service tokens

    auto subs = make_waiting_task(
        tbb::task::allocate_root(),
        [&iSubProcesses, iWait, iStreamIndex, &iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException](
            std::exception_ptr const* iPtr) mutable {
          if (iPtr) {
            auto excpt = *iPtr;
            auto delayError =
                make_waiting_task(tbb::task::allocate_root(),
                                  [iWait, excpt](std::exception_ptr const*) mutable { iWait.doneWaiting(excpt); });
            WaitingTaskHolder h(delayError);
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamEndTransitionAsync(
                  h, subProcess, iStreamIndex, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
            }
          } else {
            for (auto& subProcess : iSubProcesses) {
              subProcessDoStreamEndTransitionAsync(
                  iWait, subProcess, iStreamIndex, iPrincipal, iTS, iEventSetupImpls, cleaningUpAfterException);
            }
          }
        });

    iSchedule.processOneStreamAsync<Traits>(
        WaitingTaskHolder(subs), iStreamIndex, iPrincipal, iES, token, cleaningUpAfterException);
  }

  template <typename Traits, typename P, typename SC>
  void endStreamsTransitionAsync(WaitingTaskHolder iWait,
                                 Schedule& iSchedule,
                                 unsigned int iNStreams,
                                 P& iPrincipal,
                                 IOVSyncValue const& iTS,
                                 EventSetupImpl const& iES,
                                 std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls,
                                 ServiceToken const& iToken,
                                 SC& iSubProcesses,
                                 bool cleaningUpAfterException) {
    for (unsigned int i = 0; i < iNStreams; ++i) {
      endStreamTransitionAsync<Traits>(
          iWait, iSchedule, i, iPrincipal, iTS, iES, iEventSetupImpls, iToken, iSubProcesses, cleaningUpAfterException);
    }
  }
};  // namespace edm

#endif
