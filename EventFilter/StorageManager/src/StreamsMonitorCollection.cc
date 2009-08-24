// $Id: StreamsMonitorCollection.cc,v 1.5 2009/08/18 08:55:12 mommsen Exp $
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
_timeWindowForRecentResults(300),
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
  infoSpaceItems.push_back(std::make_pair("bandwithToDisk",  &_bandwithToDisk));
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
  _bandwithToDisk = static_cast<xdata::Double>(allStreamsVolumeStats.getValueRate(MonitoredQuantity::RECENT));
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
