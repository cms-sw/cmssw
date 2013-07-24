// $Id: DQMTopLevelFolder.cc,v 1.5 2011/04/04 16:05:37 mommsen Exp $
/// @file: DQMTopLevelFolder.cc

#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMTopLevelFolder.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"

#include "TROOT.h"

#include "toolbox/net/Utils.h"

#include <sstream>
#include <unistd.h>


namespace stor {
  
  unsigned int DQMTopLevelFolder::sentEvents_(0);
  
  DQMTopLevelFolder::DQMTopLevelFolder
  (
    const DQMKey& dqmKey,
    const QueueIDs& dqmConsumers,
    const DQMProcessingParams& dqmParams,
    DQMEventMonitorCollection& dqmEventMonColl,
    const unsigned int expectedUpdates,
    AlarmHandlerPtr alarmHandler
  ) :
  dqmKey_(dqmKey),
  dqmConsumers_(dqmConsumers),
  dqmParams_(dqmParams),
  dqmEventMonColl_(dqmEventMonColl),
  expectedUpdates_(expectedUpdates),
  alarmHandler_(alarmHandler),
  nUpdates_(0),
  mergeCount_(0),
  updateNumber_(0)
  {
    gROOT->SetBatch(kTRUE);
    dqmEventMonColl_.getNumberOfTopLevelFoldersMQ().addSample(1);
  }
  
  
  DQMTopLevelFolder::~DQMTopLevelFolder()
  {
    dqmFolders_.clear();
  }
  
  
  void DQMTopLevelFolder::addDQMEvent(const DQMEventMsgView& view)
  {
    if ( releaseTag_.empty() ) releaseTag_ = view.releaseTag();
    // A restarted EP will start counting at 0 again.
    // Thus, take the maximum of all updates we get.
    updateNumber_ = std::max(updateNumber_, view.updateNumber());
    if ( timeStamp_ == edm::Timestamp::invalidTimestamp() )
      timeStamp_ = view.timeStamp();
    else
      timeStamp_ = std::min(timeStamp_, view.timeStamp());
    mergeCount_ += std::max(1U, view.mergeCount());

    edm::StreamDQMDeserializer deserializer;
    std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
      deserializer.deserializeDQMEvent(view);
    
    addEvent(toTablePtr);
    
    ++nUpdates_;

    if (nUpdates_ > expectedUpdates_)
    {
      std::ostringstream msg;
      msg << "Received " << nUpdates_
        << " updates for top level folder " << view.topFolderName()
        << " and lumi section " << view.lumiSection()
        << " whereas only " << expectedUpdates_
        << " updates are expected.";
      XCEPT_DECLARE(exception::DQMEventProcessing,
        sentinelException, msg.str());
      alarmHandler_->notifySentinel(AlarmHandler::ERROR, sentinelException);
    }

    lastUpdate_ = utils::getCurrentTime();
    
    dqmEventMonColl_.getDQMEventSizeMQ().addSample(
      static_cast<double>(view.size()) / 0x100000
    );
  }
  
  
  bool DQMTopLevelFolder::isReady(const utils::TimePoint_t& now) const
  {
    if ( nUpdates_ == 0 ) return false;
    
    if ( nUpdates_ == expectedUpdates_ )
    {
      dqmEventMonColl_.getNumberOfCompleteUpdatesMQ().addSample(1);
      return true;
    }
    
    if ( now > lastUpdate_ + dqmParams_.readyTimeDQM_ ) return true;
    
    return false;
  }
  
  
  void DQMTopLevelFolder::addEvent(std::auto_ptr<DQMEvent::TObjectTable> toTablePtr)
  {
    for (
      DQMEvent::TObjectTable::const_iterator it = toTablePtr->begin(),
        itEnd = toTablePtr->end();
      it != itEnd; 
      ++it
    ) 
    {
      const std::string folderName = it->first;
      
      DQMFoldersMap::iterator pos = dqmFolders_.lower_bound(folderName);
      if ( pos == dqmFolders_.end() || (dqmFolders_.key_comp()(folderName, pos->first)) )
      {
        pos = dqmFolders_.insert(pos, DQMFoldersMap::value_type(
            folderName, DQMFolderPtr( new DQMFolder() )
          ));
      }
      pos->second->addObjects(it->second);
    }
  }
  
  
  bool DQMTopLevelFolder::getRecord(DQMTopLevelFolder::Record& record)
  {
    if ( nUpdates_ == 0 ) return false;
    
    record.clear();
    record.tagForEventConsumers(dqmConsumers_);
    
    // Package list of TObjects into a DQMEvent::TObjectTable
    DQMEvent::TObjectTable table;
    const size_t folderSize = populateTable(table);
    
    edm::StreamDQMSerializer serializer;
    const size_t sourceSize =
      serializer.serializeDQMEvent(table,
        dqmParams_.useCompressionDQM_,
        dqmParams_.compressionLevelDQM_);
    
    // Add space for header
    const size_t totalSize =
      sourceSize
      + sizeof(DQMEventHeader)
      + 12*sizeof(uint32_t)
      + releaseTag_.length()
      + dqmKey_.topLevelFolderName.length()
    + folderSize;
    
    DQMEventMsgBuilder builder(
      record.getBuffer(totalSize),
      totalSize,
      dqmKey_.runNumber,
      ++sentEvents_,
      timeStamp_,
      dqmKey_.lumiSection,
      updateNumber_,
      (uint32_t)serializer.adler32_chksum(),
      toolbox::net::getHostName().c_str(),
      releaseTag_,
      dqmKey_.topLevelFolderName,
      table
    ); 
    unsigned char* source = serializer.bufferPointer();
    std::copy(source,source+sourceSize, builder.eventAddress());
    builder.setEventLength(sourceSize);
    if ( dqmParams_.useCompressionDQM_ ) 
    {
      // the "compression flag" contains the uncompressed size
      builder.setCompressionFlag(serializer.currentEventSize());
    }
    else
    {
      // a size of 0 indicates no compression
      builder.setCompressionFlag(0);
    }
    builder.setMergeCount(mergeCount_);
    dqmEventMonColl_.getNumberOfUpdatesMQ().addSample(nUpdates_);
    dqmEventMonColl_.getServedDQMEventSizeMQ().addSample(
      static_cast<double>(record.totalDataSize()) / 0x100000
    );
    
    return true;
  }
  
  
  size_t DQMTopLevelFolder::populateTable(DQMEvent::TObjectTable& table) const
  {
    size_t folderSize = 0;
    
    for ( DQMFoldersMap::const_iterator it = dqmFolders_.begin(), itEnd = dqmFolders_.end();
          it != itEnd; ++it )
    {
      const std::string folderName = it->first;
      const DQMFolderPtr folder = it->second;
      
      DQMEvent::TObjectTable::iterator pos = table.lower_bound(folderName);
      if ( pos == table.end() || (table.key_comp()(folderName, pos->first)) )
      {
        std::vector<TObject*> newObjectVector;
        pos = table.insert(pos, DQMEvent::TObjectTable::value_type(folderName, newObjectVector));
        folderSize += 2*sizeof(uint32_t) + folderName.length();
      }
      folder->fillObjectVector(pos->second);
    }
    return folderSize;
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
