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
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// user include files

// forward declarations

namespace edm {
  class IOVSyncValue;
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  
  //This is code in common between beginStreamRun and beginGlobalLuminosityBlock
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder,SubProcess& iSubProcess, LuminosityBlockPrincipal& iPrincipal, IOVSyncValue const& iTS) {
    iSubProcess.doBeginLuminosityBlockAsync(std::move(iHolder),iPrincipal, iTS);
  }
  
  inline void subProcessDoGlobalBeginTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, RunPrincipal& iPrincipal, IOVSyncValue const& iTS) {
    iSubProcess.doBeginRunAsync(std::move(iHolder),iPrincipal, iTS);
  }
  
  inline void subProcessDoGlobalEndTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, LuminosityBlockPrincipal& iPrincipal, IOVSyncValue const& iTS, bool cleaningUpAfterException) {
    iSubProcess.doEndLuminosityBlockAsync(std::move(iHolder),iPrincipal, iTS,cleaningUpAfterException);
  }
  
  inline void subProcessDoGlobalEndTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, RunPrincipal& iPrincipal, IOVSyncValue const& iTS, bool cleaningUpAfterException) {
    iSubProcess.doEndRunAsync(std::move(iHolder), iPrincipal, iTS, cleaningUpAfterException);
  }

  template<typename Traits, typename P, typename SC >
  void beginGlobalTransitionAsync(WaitingTaskHolder iWait,
                                  Schedule& iSchedule,
                                  P& iPrincipal,
                                  IOVSyncValue const & iTS,
                                  EventSetupImpl const& iES,
                                  ServiceToken const& token,
                                  SC& iSubProcesses) {

    //When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs = make_waiting_task(tbb::task::allocate_root(), [&iSubProcesses, iWait,&iPrincipal,iTS](std::exception_ptr const* iPtr) mutable {
      if(iPtr) {
        auto excpt = *iPtr;
        auto delayError = make_waiting_task(tbb::task::allocate_root(), [iWait,excpt](std::exception_ptr const* ) mutable {
          iWait.doneWaiting(excpt);
        });
        WaitingTaskHolder h(delayError);
        for(auto& subProcess: iSubProcesses) {
          subProcessDoGlobalBeginTransitionAsync(h,subProcess,iPrincipal, iTS);
        }
      } else {
        for(auto& subProcess:iSubProcesses){
          subProcessDoGlobalBeginTransitionAsync(iWait,subProcess,iPrincipal, iTS);
        }
      }
    });
    
    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h),iPrincipal, iES,token);
  }

  
  template<typename Traits, typename P, typename SC >
  void endGlobalTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                P& iPrincipal,
                                IOVSyncValue const & iTS,
                                EventSetupImpl const& iES,
                                ServiceToken const& token,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException)
  {
    //When we are done processing the global for this process,
    // we need to run the global for all SubProcesses
    auto subs = make_waiting_task(tbb::task::allocate_root(), [&iSubProcesses, iWait,&iPrincipal,iTS,cleaningUpAfterException](std::exception_ptr const* iPtr) mutable {
      if(iPtr) {
        auto excpt = *iPtr;
        auto delayError = make_waiting_task(tbb::task::allocate_root(), [iWait,excpt](std::exception_ptr const* ) mutable {
          iWait.doneWaiting(excpt);
        });
        WaitingTaskHolder h(delayError);
        for(auto& subProcess: iSubProcesses){
          subProcessDoGlobalEndTransitionAsync(h,subProcess,iPrincipal, iTS,cleaningUpAfterException);
        }
      } else {
        for(auto& subProcess: iSubProcesses){
          subProcessDoGlobalEndTransitionAsync(iWait,subProcess,iPrincipal, iTS,cleaningUpAfterException);
        }
      }
    });
    
    WaitingTaskHolder h(subs);
    iSchedule.processOneGlobalAsync<Traits>(std::move(h),iPrincipal, iES,token,cleaningUpAfterException);
  }

};

#endif
