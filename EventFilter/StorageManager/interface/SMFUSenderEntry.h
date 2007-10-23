#ifndef _smfusenderentry_h_
#define _smfusenderentry_h_

#include <exception>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/Chrono.h"

namespace stor {

struct SMFUSenderEntry  // used to store each FU sender
{
  SMFUSenderEntry(const char* hltURL,
                 const char* hltClassName,
                 const unsigned int hltLocalId,
                 const unsigned int hltInstance,
                 const unsigned int hltTid,
                 const unsigned int frameCount,
                 const unsigned int numFramesToAllocate,
                 toolbox::mem::Reference *ref);
  private:
  
  char          hltURL_[MAX_I2O_SM_URLCHARS];       // FU+HLT identifiers
  char          hltClassName_[MAX_I2O_SM_URLCHARS];
  unsigned int  hltLocalId_;
  unsigned int  hltInstance_;
  unsigned int  hltTid_;
  unsigned int  registrySize_;    // size of registry in bytes once received AND copied
  bool          regAllReceived_;  // All Registry fragments are received or not
  unsigned int  totFrames_;    // number of frames in this fragment
  unsigned int  currFrames_;   // current frames received for registry
  std::vector<toolbox::mem::Reference*> frameRefs_; // vector of frame reference pointers
  std::vector<unsigned char> registryData_;
  bool          regCheckedOK_;    // Registry checked to be same as configuration
  unsigned int  connectStatus_;   // FU+HLT connection status
  double        lastLatency_;     // Latency of last frame in microseconds
  unsigned int  runNumber_;
  bool          isLocal_;         // If detected a locally sent frame chain
  // data members below are to track the data frames from this FU
  unsigned int  framesReceived_;
  unsigned int  eventsReceived_;
  unsigned int  lastEventID_;
  unsigned int  lastRunID_;
  unsigned int  lastFrameNum_;
  unsigned int  lastTotalFrameNum_;
  unsigned int  totalOutOfOrder_;
  unsigned long long  totalSizeReceived_;// For data only
  unsigned int  totalBadEvents_;   // Update meaning: include original size check?
  toolbox::Chrono chrono_;         // Keep latency for connection check
  boost::mutex entry_lock_;

  bool sameURL(const char* hltURL);
  bool sameClassName(const char* hltClassName);
  bool testCompleteFUReg();
  bool copyRegistry(toolbox::mem::Reference *head);

  public:
  
  /// returns true if frame added completes the registry
  bool addFrame(const unsigned int frameCount, const unsigned int numFrames,
                toolbox::mem::Reference *ref);
  bool update4Data(const unsigned int runNumber, const unsigned int eventNumber,
                   const unsigned int frameNum, const unsigned int totalFrames,
                   const unsigned int origdatasize);
  void setregCheckedOK(const bool status);
  void setDataStatus();
  void setrunNumber(const unsigned int run);
  void setisLocal(const bool local);
  bool regIsCopied();

  
  //double getStopWTime() const;  // more const below (nothings changes inside) didn't work due to mutex
  double getStopWTime();
  boost::shared_ptr<std::vector<char> > getvhltURL();
  boost::shared_ptr<std::vector<char> > getvhltClassName();
  unsigned int gethltLocalId() const {return hltLocalId_;}
  unsigned int gethltInstance() const {return hltInstance_;}
  unsigned int gethltTid() const {return hltTid_;}
  unsigned int getregistrySize() const {return registrySize_;}
  bool         getregAllReceived() const {return regAllReceived_;}
  unsigned int gettotFrames() const {return totFrames_;}
  unsigned int getcurrFrames() const {return currFrames_;}
  bool         getregCheckedOK() const {return regCheckedOK_;}
  unsigned int getconnectStatus() const {return connectStatus_;}
  double       getlastLatency() const {return lastLatency_;}
  unsigned int getrunNumber() const {return runNumber_;}
  bool         getisLocal() const {return isLocal_;}
  unsigned int getframesReceived() const {return framesReceived_;}
  unsigned int geteventsReceived() const {return eventsReceived_;}
  unsigned int getlastEventID() const {return lastEventID_;}
  unsigned int getlastRunID() const {return lastRunID_;}
  unsigned int getlastFrameNum() const {return lastFrameNum_;}
  unsigned int getlastTotalFrameNum() const {return lastTotalFrameNum_;}
  unsigned int gettotalOutOfOrder() const {return totalOutOfOrder_;}
  unsigned long long gettotalSizeReceived() const {return totalSizeReceived_;}
  unsigned int gettotalBadEvents() const {return totalBadEvents_;}

  bool getDataStatus();  
  char* getregistryData(); // cannot have const char* here without modifying InitMsgView ctor
  bool match(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid);
  
};
}
#endif
