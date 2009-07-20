// $Id: StreamsMonitorCollection.cc,v 1.3 2009/07/09 15:34:29 mommsen Exp $
/// @file: StreamsMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"

using namespace stor;


StreamsMonitorCollection::StreamsMonitorCollection() :
MonitorCollection(),
_timeWindowForRecentResults(300)
{
  _allStreamsFileCount.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
  _allStreamsVolume.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
  _allStreamsBandwidth.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
}


const StreamsMonitorCollection::StreamRecordPtr
StreamsMonitorCollection::getNewStreamRecord()
{
  boost::mutex::scoped_lock sl(_streamRecordsMutex);
  
  boost::shared_ptr<StreamRecord> streamRecord(new StreamsMonitorCollection::StreamRecord(this));
  streamRecord->fileCount.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
  streamRecord->volume.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
  streamRecord->bandwidth.setNewTimeWindowForRecentResults(_timeWindowForRecentResults);
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

  // These infospace items were defined in the old SM
  // infoSpaceItems.push_back(std::make_pair("namesOfStream", &_namesOfStream));
  // infoSpaceItems.push_back(std::make_pair("storedEventsInStream", &_storedEventsInStream));
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
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
