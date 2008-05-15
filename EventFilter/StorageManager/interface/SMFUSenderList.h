#ifndef _smfusenderlist_h_
#define _smfusenderlist_h_

#include <exception>
#include <list>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include "EventFilter/StorageManager/interface/SMFUSenderEntry.h"
#include "EventFilter/StorageManager/interface/SMFUSenderStats.h"

namespace stor {

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
    toolbox::mem::Reference *ref, const std::string outModName, const uint32 outModId);
  /// Update FU sender information and statistics for each data
  /// frame received, return true if this frame completes an event
  /// return -1 if problems, 1 if complete an event, 0 otherwise
  int updateFUSender4data(const char* hltURL, const char* hltClassName,
    const unsigned int hltLocalId, const unsigned int hltInstance,
    const unsigned int hltTid,
    const unsigned int runNumber, const unsigned int eventNumber,
    const unsigned int frameNum, const unsigned int totalFrames,
    const unsigned int origdatasize, const bool isLocal, const uint32 outModId);
  /// Removed a FU sender from the list when it has ended an run
  /// returns false if there was a problem finding FU sender in list
  bool removeFUSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  /// set the flag that says the registry has been checked
  void setRegCheckedOK(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName);
  /// methods for access to FU info and statistics
  char* getRegistryData(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName);
  unsigned int getRegistrySize(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, const std::string outModName);
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
    toolbox::mem::Reference *ref, const std::string outModName, const uint32 outModId);
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
