#ifndef _smfusenderlist_h_
#define _smfusenderlist_h_

#include <exception>
#include <vector>

#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/Chrono.h"

namespace stor {

struct SMFUSenderList  // used to store list of FU senders
{
  SMFUSenderList(const char* hltURL,
                 const char* hltClassName,
                 const unsigned long hltLocalId,
                 const unsigned long hltInstance,
                 const unsigned long hltTid,
                 const unsigned int numFramesToAllocate,
                 const unsigned long registrySize,
                 const char* registryData);

  char          hltURL_[MAX_I2O_SM_URLCHARS];       // FU+HLT identifiers
  char          hltClassName_[MAX_I2O_SM_URLCHARS];
  unsigned long hltLocalId_;
  unsigned long hltInstance_;
  unsigned long hltTid_;
  unsigned long registrySize_;
  bool          regAllReceived_;  // All Registry fragments are received or not
  unsigned int  totFrames_;    // number of frames in this fragment
  unsigned int  currFrames_;   // current frames received
  std::vector<toolbox::mem::Reference*> frameRefs_; // vector of frame reference pointers
  char          registryData_[2*1000*1000]; // size should be a parameter and have tests!
  bool          regCheckedOK_;    // Registry checked to be same as configuration
  unsigned int  connectStatus_;   // FU+HLT connection status
  double        lastLatency_;     // Latency of last frame in microseconds
  unsigned long runNumber_;
  bool          isLocal_;         // If detected a locally sent frame chain
  unsigned long framesReceived_;
  unsigned long eventsReceived_;
  unsigned long lastEventID_;
  unsigned long lastRunID_;
  unsigned long lastFrameNum_;
  unsigned long lastTotalFrameNum_;
  unsigned long totalOutOfOrder_;
  unsigned long totalSizeReceived_;// For data only
  unsigned long totalBadEvents_;   // Update meaning: include original size check?
  toolbox::Chrono chrono_;         // Keep latency for connection check

  //public:
  bool sameURL(const char* hltURL);
  bool sameClassName(const char* hltClassName);

};
}
#endif
