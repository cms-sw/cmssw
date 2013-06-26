// $Id: StreamsMonitorCollection.cc,v 1.22 2011/11/17 17:35:40 mommsen Exp $
/// @file: StreamsMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"


namespace stor {
  
  StreamsMonitorCollection::StreamsMonitorCollection
  (
    const utils::Duration_t& updateInterval
  ) :
  MonitorCollection(updateInterval),
  updateInterval_(updateInterval),
  timeWindowForRecentResults_(boost::posix_time::seconds(10)),
  allStreamsFileCount_(updateInterval, timeWindowForRecentResults_),
  allStreamsVolume_(updateInterval, timeWindowForRecentResults_),
  allStreamsBandwidth_(updateInterval, timeWindowForRecentResults_)
  {}
  
  
  StreamsMonitorCollection::StreamRecordPtr
  StreamsMonitorCollection::getNewStreamRecord()
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);
    
    StreamRecordPtr streamRecord(
      new StreamRecord(this,updateInterval_,timeWindowForRecentResults_)
    );
    streamRecords_.push_back(streamRecord);
    return streamRecord;
  }
  
  
  void StreamsMonitorCollection::getStreamRecords(StreamRecordList& list) const
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);

    list.clear();
    list.reserve(streamRecords_.size());
    
    for (
      StreamRecordList::const_iterator 
        it = streamRecords_.begin(), itEnd = streamRecords_.end();
      it != itEnd;
      ++it
    )
    {
      list.push_back(*it);
    }
  }
  
  
  bool StreamsMonitorCollection::getStreamRecordsForOutputModuleLabel
  (
    const std::string& label,
    StreamRecordList& list
  ) const
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);

    list.clear();
    list.reserve(streamRecords_.size());
    
    for (
      StreamRecordList::const_iterator 
        it = streamRecords_.begin(), itEnd = streamRecords_.end();
      it != itEnd;
      ++it
    )
    {
      if ( (*it)->outputModuleLabel == label )
        list.push_back(*it);
    }
    return ( ! list.empty() );
  }
  

  bool StreamsMonitorCollection::streamRecordsExist() const
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);

    return ( ! streamRecords_.empty() );
  }

  
  void StreamsMonitorCollection::StreamRecord::incrementFileCount
  (
    const uint32_t lumiSection
  )
  {
    fileCount.addSample(1);
    parentCollection->allStreamsFileCount_.addSample(1);
    ++fileCountPerLS[lumiSection];
  }
  
  
  void StreamsMonitorCollection::StreamRecord::addSizeInBytes(double size)
  {
    size = size / (1024 * 1024);
    volume.addSample(size);
    parentCollection->allStreamsVolume_.addSample(size);
  }
  
  
  bool StreamsMonitorCollection::StreamRecord::reportLumiSectionInfo
  (
    const uint32_t& lumiSection,
    std::string& str
  )
  {
    std::ostringstream msg;
    if (str.empty())
    {
      msg << "LS:" << lumiSection;
    }
    
    unsigned int count = 0;
    FileCountPerLumiSectionMap::iterator pos = fileCountPerLS.find(lumiSection);
    if ( pos != fileCountPerLS.end() )
    {
      count = pos->second;
      fileCountPerLS.erase(pos);
    }
    msg << "\t" << streamName << ":" << count;
    str += msg.str();

    return (count>0);
  }
  
  
  void StreamsMonitorCollection::reportAllLumiSectionInfos
  (
    DbFileHandlerPtr dbFileHandler,
    EndOfRunReportPtr endOfRunReport
  )
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);

    UnreportedLS unreportedLS;
    getListOfAllUnreportedLS(unreportedLS);
    
    for (UnreportedLS::const_iterator it = unreportedLS.begin(),
           itEnd = unreportedLS.end(); it != itEnd; ++it)
    {
      std::string lsEntry;
      bool filesWritten = false;

      for (StreamRecordList::const_iterator 
             stream = streamRecords_.begin(),
             streamEnd = streamRecords_.end();
           stream != streamEnd;
           ++stream)
      {
        if ( (*stream)->reportLumiSectionInfo((*it), lsEntry) )
          filesWritten = true;
      }
      lsEntry += "\tEoLS:0";
      dbFileHandler->write(lsEntry);

      if (filesWritten) ++(endOfRunReport->lsCountWithFiles);
      endOfRunReport->updateLatestWrittenLumiSection(*it);
    }
  }
  
  
  void StreamsMonitorCollection::getListOfAllUnreportedLS(UnreportedLS& unreportedLS)
  {
    // Have to loop over all streams as not every stream
    // might have got an event for a given lumi section
    for (StreamRecordList::const_iterator 
           stream = streamRecords_.begin(),
           streamEnd = streamRecords_.end();
         stream != streamEnd;
         ++stream)
    {
      for (StreamRecord::FileCountPerLumiSectionMap::const_iterator
             lscount = (*stream)->fileCountPerLS.begin(),
             lscountEnd = (*stream)->fileCountPerLS.end();
           lscount != lscountEnd; ++lscount)
      {
        unreportedLS.insert(lscount->first);
      }
    }
  }
  
  
  void StreamsMonitorCollection::do_calculateStatistics()
  {
    MonitoredQuantity::Stats stats;
    
    allStreamsFileCount_.calculateStatistics();
    allStreamsVolume_.calculateStatistics();
    allStreamsVolume_.getStats(stats);
    bool samplingHasStarted = (stats.getSampleCount() > 0);
    if (samplingHasStarted) {
      allStreamsBandwidth_.addSample(stats.getLastValueRate());
    }
    allStreamsBandwidth_.calculateStatistics();
    
    
    boost::mutex::scoped_lock sl(streamRecordsMutex_);
    
    for (
      StreamRecordList::const_iterator 
        it = streamRecords_.begin(), itEnd = streamRecords_.end();
      it != itEnd;
      ++it
    ) 
    {
      (*it)->fileCount.calculateStatistics();
      (*it)->volume.calculateStatistics();
      (*it)->volume.getStats(stats);
      if (samplingHasStarted) {
        (*it)->bandwidth.addSample(stats.getLastValueRate());
      }
      (*it)->bandwidth.calculateStatistics();
    }
  }
  
  
  void StreamsMonitorCollection::do_appendInfoSpaceItems
  (
    InfoSpaceItems& infoSpaceItems
  )
  {
    infoSpaceItems.push_back(std::make_pair("storedEvents",  &storedEvents_));
    infoSpaceItems.push_back(std::make_pair("storedVolume",  &storedVolume_));
    infoSpaceItems.push_back(std::make_pair("bandwidthToDisk",  &bandwidthToDisk_));
    infoSpaceItems.push_back(std::make_pair("streamNames",  &streamNames_));
    infoSpaceItems.push_back(std::make_pair("eventsPerStream",  &eventsPerStream_));
    infoSpaceItems.push_back(std::make_pair("ratePerStream",  &ratePerStream_));
    infoSpaceItems.push_back(std::make_pair("bandwidthPerStream",  &bandwidthPerStream_));
  }
  
  
  void StreamsMonitorCollection::do_reset()
  {
    allStreamsFileCount_.reset();
    allStreamsVolume_.reset();
    allStreamsBandwidth_.reset();
    
    boost::mutex::scoped_lock sl(streamRecordsMutex_);
    streamRecords_.clear();
  }
  
  
  void StreamsMonitorCollection::do_updateInfoSpaceItems()
  {
    MonitoredQuantity::Stats allStreamsVolumeStats;
    allStreamsVolume_.getStats(allStreamsVolumeStats);
    
    storedEvents_ = static_cast<xdata::UnsignedInteger32>(
      allStreamsVolumeStats.getSampleCount()
    );
    storedVolume_ = static_cast<xdata::Double>(
      allStreamsVolumeStats.getValueSum()
    );
    bandwidthToDisk_ = static_cast<xdata::Double>(
      allStreamsVolumeStats.getValueRate(MonitoredQuantity::RECENT)
    );
    
    boost::mutex::scoped_lock sl(streamRecordsMutex_);
    
    streamNames_.clear();
    eventsPerStream_.clear();
    ratePerStream_.clear();
    bandwidthPerStream_.clear();
    
    streamNames_.reserve(streamRecords_.size());
    eventsPerStream_.reserve(streamRecords_.size());
    ratePerStream_.reserve(streamRecords_.size());
    bandwidthPerStream_.reserve(streamRecords_.size());
    
    for (
      StreamRecordList::const_iterator
        it = streamRecords_.begin(), itEnd = streamRecords_.end();
      it != itEnd;
      ++it
    )
    {
      MonitoredQuantity::Stats streamVolumeStats;
      (*it)->volume.getStats(streamVolumeStats);
      MonitoredQuantity::Stats streamBandwidthStats;
      (*it)->bandwidth.getStats(streamBandwidthStats);
      
      streamNames_.push_back(
        static_cast<xdata::String>( (*it)->streamName )
      );
      
      eventsPerStream_.push_back(
        static_cast<xdata::UnsignedInteger32>(
          streamVolumeStats.getSampleCount(MonitoredQuantity::FULL)
        )
      );
      
      ratePerStream_.push_back(
        static_cast<xdata::Double>(
          streamVolumeStats.getSampleRate(MonitoredQuantity::RECENT)
        )
      );
      
      bandwidthPerStream_.push_back(
        static_cast<xdata::Double>(
          streamBandwidthStats.getValueRate(MonitoredQuantity::RECENT)
        )
      );
    }
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
