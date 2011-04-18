// $Id: StreamsMonitorCollection.cc,v 1.16 2011/04/14 12:52:48 mommsen Exp $
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
  timeWindowForRecentResults_(boost::posix_time::seconds(30)),
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
  
  
  void StreamsMonitorCollection::StreamRecord::reportLumiSectionInfo
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
  }
  
  
  void StreamsMonitorCollection::reportAllLumiSectionInfos(DbFileHandlerPtr dbFileHandler)
  {
    boost::mutex::scoped_lock sl(streamRecordsMutex_);
    
    UnreportedLS unreportedLS;
    getListOfAllUnreportedLS(unreportedLS);
    
    for (UnreportedLS::const_iterator it = unreportedLS.begin(),
           itEnd = unreportedLS.end(); it != itEnd; ++it)
    {
      std::string lsEntry;
      for (StreamRecordList::const_iterator 
             stream = streamRecords_.begin(),
             streamEnd = streamRecords_.end();
           stream != streamEnd;
           ++stream)
      {
        (*stream)->reportLumiSectionInfo((*it), lsEntry);
      }
      dbFileHandler->write(lsEntry);
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
    
    const size_t statsCount = streamRecords_.size();
    const size_t infospaceCount = streamNames_.size();

    if ( statsCount != infospaceCount )
    {
      streamNames_.resize(statsCount);
      eventsPerStream_.resize(statsCount);
      ratePerStream_.resize(statsCount);
      bandwidthPerStream_.resize(statsCount);
    }

    for (size_t i=0; i < statsCount; ++i)
    {
      MonitoredQuantity::Stats streamVolumeStats;
      streamRecords_.at(i)->volume.getStats(streamVolumeStats);
      MonitoredQuantity::Stats streamBandwidthStats;
      streamRecords_.at(i)->bandwidth.getStats(streamBandwidthStats);

      streamNames_.at(i) = static_cast<xdata::String>(streamRecords_.at(i)->streamName);
      eventsPerStream_.at(i) = static_cast<xdata::UnsignedInteger32>(
        streamVolumeStats.getSampleCount(MonitoredQuantity::FULL)
      );
      ratePerStream_.at(i) = static_cast<xdata::Double>(
        streamVolumeStats.getSampleRate(MonitoredQuantity::RECENT)
      );
      bandwidthPerStream_.at(i) = static_cast<xdata::Double>(
        streamBandwidthStats.getValueRate(MonitoredQuantity::RECENT)
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
