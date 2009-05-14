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

class SMFUSenderList  //< list of data senders with thread-safe access
{
  public:

  SMFUSenderList();

  virtual ~SMFUSenderList(){}

  // following method uses the list lock

  void clear();
  /// number of registered data senders
  unsigned int size();
  unsigned int numberOfRB() const { return numberOfRB_;}
  unsigned int numberOfOM() const { return numberOfOM_;}
  unsigned int numberOfFU();
  /// register INIT message frame from data sender
  /// Creates an entry or update one for subsequent frames
  ///  containing additional fragments
  /// return -1 if problems, 1 if registry is completed, 0 otherwise
  int registerDataSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    const uint32 regSize, const std::string outModName, 
    const uint32 outModId, const uint32 rbBufferID);
  /// Update data sender information and statistics for each data
  /// frame received, return true if this frame completes an event
  /// return -1 if problems, 1 if complete an event, 0 otherwise
  int updateSender4data(const char* hltURL, const char* hltClassName,
    const unsigned int hltLocalId, const unsigned int hltInstance,
    const unsigned int hltTid,
    const unsigned int runNumber, const unsigned int eventNumber,
    const unsigned int frameNum, const unsigned int totalFrames,
    const unsigned int origdatasize, const bool isLocal, const uint32 outModId);
  /// Removed a data sender from the list when it has ended an run
  /// returns false if there was a problem finding the sender in list
  bool removeDataSender(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid);
  /// methods for access to sender info and statistics
  unsigned int getRegistrySize(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid, 
    const std::string outModName, const uint32 rbBufferID);
  /// provide access to (self-consistent) statistics
  std::vector<boost::shared_ptr<SMFUSenderStats> > getSenderStats();

  private:

  // following methods do not use the list lock internally

  boost::shared_ptr<stor::SMFUSenderEntry> findFirstEntry(const char* hltURL, 
    const char* hltClassName, const unsigned int hltLocalId, 
    const unsigned int hltInstance, const unsigned int hltTid);
  boost::shared_ptr<stor::SMFUSenderEntry> findEntry(const char* hltURL, 
    const char* hltClassName, const unsigned int hltLocalId, 
    const unsigned int hltInstance, const unsigned int hltTid,
    const uint32 rbBufferID, const std::string outModName);
  boost::shared_ptr<stor::SMFUSenderEntry> findFirstEntry(const char* hltURL, 
    const char* hltClassName, const unsigned int hltLocalId, 
    const unsigned int hltInstance, const unsigned int hltTid,
    const std::string outModName);
  boost::shared_ptr<stor::SMFUSenderEntry> findFirstEntry(const char* hltURL, 
    const char* hltClassName, const unsigned int hltLocalId, 
    const unsigned int hltInstance, const unsigned int hltTid,
    const uint32 outModId);
  boost::shared_ptr<stor::SMFUSenderEntry> addEntry(const char* hltURL,
    const char* hltClassName, const unsigned int hltLocalId,
    const unsigned int hltInstance, const unsigned int hltTid,
    const unsigned int frameCount, const unsigned int numFrames,
    const uint32 regSize, const std::string outModName, 
    const uint32 outModId, const uint32 rbBufferID);
/*
  bool eraseFirstFUEntry(const char* hltURL, const char* hltClassName, 
                  const unsigned int hltLocalId,
                  const unsigned int hltInstance, 
                  const unsigned int hltTid);
*/

  // for large numbers of data senders we should change this later to a map
  std::list<boost::shared_ptr<stor::SMFUSenderEntry> > senderlist_;
  unsigned int numberOfRB_;
  unsigned int numberOfOM_;
  boost::mutex list_lock_;
  
};
}
#endif
