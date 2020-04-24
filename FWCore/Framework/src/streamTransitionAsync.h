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
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/SubProcess.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// user include files

// forward declarations

namespace edm {
  class IOVSyncValue;
  class EventSetup;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  
  //This is code in common between beginStreamRun and beginStreamLuminosityBlock
  inline void subProcessDoStreamBeginTransitionAsync(WaitingTaskHolder iHolder,SubProcess& iSubProcess, unsigned int i, LuminosityBlockPrincipal& iPrincipal, IOVSyncValue const& iTS) {
    iSubProcess.doStreamBeginLuminosityBlockAsync(std::move(iHolder),i,iPrincipal, iTS);
  }
  
  inline void subProcessDoStreamBeginTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, unsigned int i, RunPrincipal& iPrincipal, IOVSyncValue const& iTS) {
    iSubProcess.doStreamBeginRunAsync(std::move(iHolder),i,iPrincipal, iTS);
  }
  
  inline void subProcessDoStreamEndTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, unsigned int i, LuminosityBlockPrincipal& iPrincipal, IOVSyncValue const& iTS, bool cleaningUpAfterException) {
    iSubProcess.doStreamEndLuminosityBlockAsync(std::move(iHolder),i,iPrincipal, iTS,cleaningUpAfterException);
  }
  
  inline void subProcessDoStreamEndTransitionAsync(WaitingTaskHolder iHolder, SubProcess& iSubProcess, unsigned int i, RunPrincipal& iPrincipal, IOVSyncValue const& iTS, bool cleaningUpAfterException) {
    iSubProcess.doStreamEndRunAsync(std::move(iHolder), i ,iPrincipal, iTS, cleaningUpAfterException);
  }


  template<typename Traits, typename P, typename SC >
  void beginStreamTransitionAsync(WaitingTaskHolder iWait,
                                  Schedule& iSchedule,
                                  unsigned int iStreamIndex,
                                  P& iPrincipal,
                                  IOVSyncValue const & iTS,
                                  EventSetup const& iES,
                                  SC& iSubProcesses) {
    ServiceToken token = ServiceRegistry::instance().presentToken();

    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    auto subs = make_waiting_task(tbb::task::allocate_root(), [&iSubProcesses, iWait,iStreamIndex,&iPrincipal,iTS,token](std::exception_ptr const* iPtr) mutable {
      if(iPtr) {
        iWait.doneWaiting(*iPtr);
        return;
      }
      ServiceRegistry::Operate op(token);
      for_all(iSubProcesses, [&iWait,iStreamIndex, &iPrincipal, iTS](auto& subProcess){ subProcessDoStreamBeginTransitionAsync(iWait,subProcess,iStreamIndex,iPrincipal, iTS); });
    });
    
    WaitingTaskHolder h(subs);
    iSchedule.processOneStreamAsync<Traits>(std::move(h), iStreamIndex,iPrincipal, iES);
  }

  
  template<typename Traits, typename P, typename SC >
  void beginStreamsTransitionAsync(WaitingTask* iWait,
                                  Schedule& iSchedule,
                                  unsigned int iNStreams,
                                  P& iPrincipal,
                                  IOVSyncValue const & iTS,
                                  EventSetup const& iES,
                                  SC& iSubProcesses)
  {
    WaitingTaskHolder holdUntilAllStreamsCalled(iWait);
    for(unsigned int i=0; i<iNStreams;++i) {
      beginStreamTransitionAsync<Traits>(WaitingTaskHolder(iWait), iSchedule,i,iPrincipal,iTS,iES,iSubProcesses);
    }
  }
  
  template<typename Traits, typename P, typename SC >
  void endStreamTransitionAsync(WaitingTaskHolder iWait,
                                Schedule& iSchedule,
                                unsigned int iStreamIndex,
                                P& iPrincipal,
                                IOVSyncValue const & iTS,
                                EventSetup const& iES,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException)
  {
    ServiceToken token = ServiceRegistry::instance().presentToken();
    
    //When we are done processing the stream for this process,
    // we need to run the stream for all SubProcesses
    auto subs = make_waiting_task(tbb::task::allocate_root(), [&iSubProcesses, iWait,iStreamIndex,&iPrincipal,iTS,token,cleaningUpAfterException](std::exception_ptr const* iPtr) mutable {
      if(iPtr) {
        iWait.doneWaiting(*iPtr);
        return;
      }
      ServiceRegistry::Operate op(token);
      for_all(iSubProcesses, [&iWait,iStreamIndex, &iPrincipal, iTS,cleaningUpAfterException](auto& subProcess){
        subProcessDoStreamEndTransitionAsync(iWait,subProcess,iStreamIndex,iPrincipal, iTS,cleaningUpAfterException); });
    });
    
    WaitingTaskHolder h(subs);
    iSchedule.processOneStreamAsync<Traits>(std::move(h), iStreamIndex,iPrincipal, iES,cleaningUpAfterException);
      
    
  }

  template<typename Traits, typename P, typename SC >
  void endStreamsTransitionAsync(WaitingTask* iWait,
                                Schedule& iSchedule,
                                unsigned int iNStreams,
                                P& iPrincipal,
                                IOVSyncValue const & iTS,
                                EventSetup const& iES,
                                SC& iSubProcesses,
                                bool cleaningUpAfterException)
  {
    WaitingTaskHolder holdUntilAllStreamsCalled(iWait);
    for(unsigned int i=0; i<iNStreams;++i) {
      endStreamTransitionAsync<Traits>(WaitingTaskHolder(iWait),
                                       iSchedule,i,
                                       iPrincipal,iTS,iES,
                                       iSubProcesses,cleaningUpAfterException);
    }
  }
};

#endif
