// $Id: StreamsMonitorCollection.cc,v 1.7 2009/12/16 14:44:49 mommsen Exp $
/// @file: StreamsMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"

using namespace stor;


StreamsMonitorCollection::StreamsMonitorCollection(const utils::duration_t& updateInterval) :
MonitorCollection(updateInterval),
_updateInterval(updateInterval),
_timeWindowForRecentResults(30),
_allStreamsFileCount(updateInterval, _timeWindowForRecentResults),
_allStreamsVolume(updateInterval, _timeWindowForRecentResults),
_allStreamsBandwidth(updateInterval, _timeWindowForRecentResults)
{}


const StreamsMonitorCollection::StreamRecordPtr
StreamsMonitorCollection::getNewStreamRecord()
{
  boost::mutex::scoped_lock sl(_streamRecordsMutex);

  StreamRecordPtr streamRecord(
    new StreamRecord(this,_updateInterval,_timeWindowForRecentResults)
  );
  _streamRecords.push_back(streamRecord);
  return streamRecord;
}


void StreamsMonitorCollection::StreamRecord::incrementFileCount()
{
  fileCount.addSample(1);
  parentCollection->_allStreamsFileCount.addSample(1);
}


void StreamsMonitorCollection::StreamRecord::addSizeInBytes(double size)
{
  size = size / (1024 * 1024);
  volume.addSample(size);
  parentCollection->_allStreamsVolume.addSample(size);
}


void StreamsMonitorCollection::do_calculateStatistics()
{
  MonitoredQuantity::Stats stats;

  _allStreamsFileCount.calculateStatistics();
  _allStreamsVolume.calculateStatistics();
  _allStreamsVolume.getStats(stats);
  bool samplingHasStarted = (stats.getSampleCount() > 0);
  if (samplingHasStarted) {
    _allStreamsBandwidth.addSample(stats.getLastValueRate());
  }
  _allStreamsBandwidth.calculateStatistics();


  boost::mutex::scoped_lock sl(_streamRecordsMutex);

  for (
    StreamRecordList::const_iterator 
      it = _streamRecords.begin(), itEnd = _streamRecords.end();
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


void StreamsMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("storedEvents",  &_storedEvents));
  infoSpaceItems.push_back(std::make_pair("storedVolume",  &_storedVolume));
  // Leave parameter with typo (bandwith) for backwards compatibility
  infoSpaceItems.push_back(std::make_pair("bandwithToDisk",  &_bandwidthToDisk));
  infoSpaceItems.push_back(std::make_pair("bandwidthToDisk",  &_bandwidthToDisk));
  infoSpaceItems.push_back(std::make_pair("streamNames",  &_streamNames));
  infoSpaceItems.push_back(std::make_pair("eventsPerStream",  &_eventsPerStream));
  infoSpaceItems.push_back(std::make_pair("ratePerStream",  &_ratePerStream));
  infoSpaceItems.push_back(std::make_pair("bandwidthPerStream",  &_bandwidthPerStream));
}


void StreamsMonitorCollection::do_reset()
{
  _allStreamsFileCount.reset();
  _allStreamsVolume.reset();
  _allStreamsBandwidth.reset();

  boost::mutex::scoped_lock sl(_streamRecordsMutex);
  _streamRecords.clear();
}


void StreamsMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats allStreamsVolumeStats;
  _allStreamsVolume.getStats(allStreamsVolumeStats);
  
  _storedEvents = static_cast<xdata::UnsignedInteger32>(allStreamsVolumeStats.getSampleCount());
  _storedVolume = static_cast<xdata::Double>(allStreamsVolumeStats.getValueSum());
  _bandwidthToDisk = static_cast<xdata::Double>(allStreamsVolumeStats.getValueRate(MonitoredQuantity::RECENT));

  _streamNames.clear();
  _eventsPerStream.clear();
  _ratePerStream.clear();
  _bandwidthPerStream.clear();

  _streamNames.reserve(_streamRecords.size());
  _eventsPerStream.reserve(_streamRecords.size());
  _ratePerStream.reserve(_streamRecords.size());
  _bandwidthPerStream.reserve(_streamRecords.size());

   for (
    StreamRecordList::const_iterator
      it = _streamRecords.begin(), itEnd = _streamRecords.end();
    it != itEnd;
    ++it
  )
  {
    MonitoredQuantity::Stats streamVolumeStats;
    (*it)->volume.getStats(streamVolumeStats);
    MonitoredQuantity::Stats streamBandwidthStats;
    (*it)->bandwidth.getStats(streamBandwidthStats);
    
    _streamNames.push_back(
      static_cast<xdata::String>( (*it)->streamName )
    );
    
    _eventsPerStream.push_back(
      static_cast<xdata::UnsignedInteger32>( streamVolumeStats.getSampleCount(MonitoredQuantity::FULL) )
    );
   
    _ratePerStream.push_back(
      static_cast<xdata::Double>( streamVolumeStats.getSampleRate(MonitoredQuantity::RECENT) )
    );

    _bandwidthPerStream.push_back(
      static_cast<xdata::Double>( streamBandwidthStats.getValueRate(MonitoredQuantity::RECENT) )
    );
  }
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
