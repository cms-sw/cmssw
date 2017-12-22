#ifndef FWCore_Framework_LuminosityBlockProcessingStatus_h
#define FWCore_Framework_LuminosityBlockProcessingStatus_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     LuminosityBlockProcessingStatus
// 
/**\class LuminosityBlockProcessingStatus LuminosityBlockProcessingStatus.h "LuminosityBlockProcessingStatus.h"

 Description: Keep status information about one LuminosityBlock transition

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 19 Dec 2017 14:24:57 GMT
//

// system include files
#include <memory>
#include <atomic>

// user include files
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"

// forward declarations
namespace edm {
#if !defined(TEST_NO_FWD_DECL)
  class EventProcessor;
  class LuminosityBlockPrincipal;
  class WaitingTaskHolder;
  class LuminosityBlockProcessingStatus;
  void globalEndLumiAsync(edm::WaitingTaskHolder iTask, std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus);
#endif

class LuminosityBlockProcessingStatus
{

  public:
  friend void globalEndLumiAsync(WaitingTaskHolder iTask, std::shared_ptr<LuminosityBlockProcessingStatus> iLumiStatus);

  
  LuminosityBlockProcessingStatus(EventProcessor* iEP, unsigned int iNStreams):
  eventProcessor_(iEP), nStreamsStillProcessingLumi_(iNStreams) {}

  std::shared_ptr<LuminosityBlockPrincipal>& lumiPrincipal() { return lumiPrincipal_;}

  void setResumer( LimitedTaskQueue::Resumer iResumer ) {
    globalLumiQueueResumer_ = std::move(iResumer);
  }
  void resumeGlobalLumiQueue() {
    globalLumiQueueResumer_.resume();
  }
  
  bool streamFinishedLumi() {
    return 0 == (--nStreamsStillProcessingLumi_);
  }
  
  bool wasEventProcessingStopped() const { return stopProcessingEvents_;}
  void stopProcessingEvents() { stopProcessingEvents_ = true;}
  void startProcessingEvents() { stopProcessingEvents_ = false;}
  
  bool isLumiEnding() const {return lumiEnding_;}
  void endLumi() { lumiEnding_ = true;}
  
  bool continuingLumi() const { return continuingLumi_;}
  void haveContinuedLumi() { continuingLumi_ = false; }
  void needToContinueLumi() { continuingLumi_ = true;}

  bool haveStartedNextLumi() const { return startedNextLumi_;}
  void startNextLumi() { startedNextLumi_ = true;}
  
  private:
  // ---------- member data --------------------------------
  std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;
  LimitedTaskQueue::Resumer globalLumiQueueResumer_;
  EventProcessor* eventProcessor_ = nullptr;
  std::atomic<unsigned int> nStreamsStillProcessingLumi_{0}; //read/write as streams finish lumi so must be atomic
  bool stopProcessingEvents_{false}; //read/write in m_sourceQueue OR from main thread when no tasks running
  bool lumiEnding_{false}; //read/write in m_sourceQueue NOTE: This is a useful cache instead of recalculating each call
  bool continuingLumi_{false}; //read/write in m_sourceQueue OR from main thread when no tasks running
  bool startedNextLumi_{false}; //read/write in m_sourceQueue


};
}

#endif
