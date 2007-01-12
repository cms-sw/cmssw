#ifndef _smfusenderlist_h_
#define _smfusenderlist_h_

#include <exception>
#include <list>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "EventFilter/StorageManager/interface/SMFUSenderEntry.h"

namespace stor {

// should be using this same (actually slightly modified) struct in SMFUSenderEntry
struct SMFUSenderStats // for FU sender statistics (from SMFUSenderEntry)
{
  SMFUSenderStats(boost::shared_ptr<std::vector<char> > hltURL,
                  boost::shared_ptr<std::vector<char> >  hltClassName,
                  unsigned int  hltLocalId,
                  unsigned int  hltInstance,
                  unsigned int  hltTid,
                  unsigned int  registrySize,
                  bool          regAllReceived,
                  unsigned int  totFrames,
                  unsigned int  currFrames,
                  bool          regCheckedOK,
                  unsigned int  connectStatus,
                  double        lastLatency,
                  unsigned int  runNumber,
                  bool          isLocal,
                  unsigned int  framesReceived,
                  unsigned int  eventsReceived,
                  unsigned int  lastEventID,
                  unsigned int  lastRunID,
                  unsigned int  lastFrameNum,
                  unsigned int  lastTotalFrameNum,
                  unsigned int  totalOutOfOrder,
                  unsigned long long  totalSizeReceived,
                  unsigned int  totalBadEvents,
                  double        timewaited);

  boost::shared_ptr<std::vector<char> >  hltURL_;       // FU+HLT identifiers
  boost::shared_ptr<std::vector<char> >  hltClassName_;
  unsigned int  hltLocalId_;
  unsigned int  hltInstance_;
  unsigned int  hltTid_;
  unsigned int  registrySize_;    // size of registry in bytes once received AND copied
  bool          regAllReceived_;  // All Registry fragments are received or not
  unsigned int  totFrames_;    // number of frames in this fragment
  unsigned int  currFrames_;   // current frames received for registry
  bool          regCheckedOK_;    // Registry checked to be same as configuration
  unsigned int  connectStatus_;   // FU+HLT connection status
  double        lastLatency_;     // Latency of last frame in microseconds
  unsigned int  runNumber_;
  bool          isLocal_;         // If detected a locally sent frame chain
  unsigned int  framesReceived_;
  unsigned int  eventsReceived_;
  unsigned int  lastEventID_;
  unsigned int  lastRunID_;
  unsigned int  lastFrameNum_;
  unsigned int  lastTotalFrameNum_;
  unsigned int  totalOutOfOrder_;
  unsigned long long  totalSizeReceived_;// For data only
  unsigned int  totalBadEvents_;   // Update meaning: include original size check?
  double        timeWaited_; // time since last frame in microseconds
};

SMFUSenderStats::SMFUSenderStats(boost::shared_ptr<std::vector<char> > hltURL,
                  boost::shared_ptr<std::vector<char> > hltClassName,
                  unsigned int  hltLocalId,
                  unsigned int  hltInstance,
                  unsigned int  hltTid,
                  unsigned int  registrySize,
                  bool          regAllReceived,
                  unsigned int  totFrames,
                  unsigned int  currFrames,
                  bool          regCheckedOK,
                  unsigned int  connectStatus,
                  double        lastLatency,
                  unsigned int  runNumber,
                  bool          isLocal,
                  unsigned int  framesReceived,
                  unsigned int  eventsReceived,
                  unsigned int  lastEventID,
                  unsigned int  lastRunID,
                  unsigned int  lastFrameNum,
                  unsigned int  lastTotalFrameNum,
                  unsigned int  totalOutOfOrder,
                  unsigned long long  totalSizeReceived,
                  unsigned int  totalBadEvents,
                  double        timewaited):
  hltURL_(hltURL), // is this right?
  hltClassName_(hltClassName),
  hltLocalId_(hltLocalId),
  hltInstance_(hltInstance),
  hltTid_(hltTid),
  registrySize_(registrySize),
  regAllReceived_(regAllReceived),
  totFrames_(totFrames),
  currFrames_(currFrames),
  regCheckedOK_(regCheckedOK),
  connectStatus_(connectStatus),
  lastLatency_(lastLatency),
  runNumber_(runNumber),
  isLocal_(isLocal),
  framesReceived_(framesReceived),
  eventsReceived_(eventsReceived),
  lastEventID_(lastEventID),
  lastRunID_(lastRunID),
  lastFrameNum_(lastFrameNum),
  lastTotalFrameNum_(lastTotalFrameNum),
  totalOutOfOrder_(totalOutOfOrder),
  totalSizeReceived_(totalSizeReceived),
  totalBadEvents_(totalBadEvents),
  timeWaited_(timewaited)
{
}

class SMFUSenderList  //< list of FU senders with thread-safe access
{
  public:

  SMFUSenderList();

  virtual ~SMFUSenderList(){}

  // following method uses the list lock

  /// number of registered FU senders
  unsigned int size();
  /// register INIT message frame from FU sender
  /// Creates an entry or update one for subsequent frames
  ///  containing additional fragments
  /// return -1 if problems, 1 if registry is completed, 0 otherwise
  int registerFUSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    toolbox::mem::Reference *ref);
  /// Update FU sender information and statistics for each data
  /// frame received, return true if this frame completes an event
  /// return -1 if problems, 1 if complete an event, 0 otherwise
  int updateFUSender4data(const char* hltURL, const char* hltClassName,
    const unsigned int hltLocalId, const unsigned int hltInstance,
    const unsigned int hltTid,
    const unsigned int runNumber, const unsigned int eventNumber,
    const unsigned int frameNum, const unsigned int totalFrames,
    const unsigned int origdatasize, const bool isLocal);
  /// Removed a FU sender from the list when it has ended an run
  /// returns false if there was a problem finding FU sender in list
  bool removeFUSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  /// set the flag that says the registry has been checked
  void setRegCheckedOK(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  /// methods for access to FU info and statistics
  char* getRegistryData(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  unsigned int getRegistrySize(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  /// provide access to (self-consistent) statistics
  std::vector<boost::shared_ptr<SMFUSenderStats> > getFUSenderStats();

  private:

  // following methods do not use the list lock internally

  boost::shared_ptr<stor::SMFUSenderEntry> findEntry(const char* hltURL, 
    const char* hltClassName, const unsigned int hltLocalId, 
    const unsigned int hltInstance, const unsigned int hltTid);
  boost::shared_ptr<stor::SMFUSenderEntry> addEntry(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    toolbox::mem::Reference *ref);
  bool eraseEntry(const char* hltURL, const char* hltClassName, 
                  const unsigned int hltLocalId,
                  const unsigned int hltInstance, 
                  const unsigned int hltTid);

  // for large numbers of FU senders we should change this later to a map
  std::list<boost::shared_ptr<stor::SMFUSenderEntry> > fulist_;
  boost::mutex list_lock_;
  
};
}
#endif
