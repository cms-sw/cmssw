#ifndef _smfusenderentry_h_
#define _smfusenderentry_h_

#include <exception>
#include <vector>

#include "boost/thread/thread.hpp"

#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"
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
  char          registryData_[2*1000*1000]; // change this to a vector<char>
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
  unsigned int  totalSizeReceived_;// For data only
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
  unsigned int gettotFrames();
  unsigned int getcurrFrames();
  bool getDataStatus();  
  unsigned int getrunNumber();
  char* getregistryData(); // cannot have const char* here without modifying InitMsgView ctor
  unsigned int getregistrySize();
  bool match(const char* hltURL, const char* hltClassName, 
                             const unsigned int hltLocalId,
                             const unsigned int hltInstance, 
                             const unsigned int hltTid);
  
};
}
#endif
