// $Id: StreamsMonitorCollection.cc,v 1.12 2010/03/25 09:14:03 mommsen Exp $
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


void StreamsMonitorCollection::StreamRecord::incrementFileCount(const uint32_t lumiSection)
{
  fileCount.addSample(1);
  parentCollection->_allStreamsFileCount.addSample(1);
  ++fileCountPerLS[lumiSection];
}


void StreamsMonitorCollection::StreamRecord::addSizeInBytes(double size)
{
  size = size / (1024 * 1024);
  volume.addSample(size);
  parentCollection->_allStreamsVolume.addSample(size);
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
  boost::mutex::scoped_lock sl(_streamRecordsMutex);

  UnreportedLS unreportedLS;
  getListOfAllUnreportedLS(unreportedLS);

  for (UnreportedLS::const_iterator it = unreportedLS.begin(),
         itEnd = unreportedLS.end(); it != itEnd; ++it)
  {
    std::string lsEntry;
    for (StreamRecordList::const_iterator 
           stream = _streamRecords.begin(),
           streamEnd = _streamRecords.end();
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
         stream = _streamRecords.begin(),
         streamEnd = _streamRecords.end();
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

  boost::mutex::scoped_lock sl(_streamRecordsMutex);

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
