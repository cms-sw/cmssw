// $Id: DQMEventMonitorCollection.cc,v 1.13 2011/04/04 12:03:30 mommsen Exp $
/// @file: DQMEventMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"

namespace stor
{
  
  DQMEventMonitorCollection::DQMEventMonitorCollection(const utils::Duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  droppedDQMEventCounts_(updateInterval, boost::posix_time::seconds(300)),
  dqmEventSizes_(updateInterval, boost::posix_time::seconds(300)),
  servedDQMEventSizes_(updateInterval, boost::posix_time::seconds(300)),
  writtenDQMEventSizes_(updateInterval, boost::posix_time::seconds(300)),
  dqmEventBandwidth_(updateInterval, boost::posix_time::seconds(300)),
  servedDQMEventBandwidth_(updateInterval, boost::posix_time::seconds(300)),
  writtenDQMEventBandwidth_(updateInterval, boost::posix_time::seconds(300)),
  numberOfTopLevelFolders_(updateInterval, boost::posix_time::seconds(300)),
  numberOfUpdates_(updateInterval, boost::posix_time::seconds(300)),
  numberOfWrittenTopLevelFolders_(updateInterval, boost::posix_time::seconds(300)),
  numberOfCompleteUpdates_(updateInterval, boost::posix_time::seconds(300))
  {}
  
  
  void DQMEventMonitorCollection::getStats(DQMEventStats& stats) const
  {
    getDroppedDQMEventCountsMQ().getStats(stats.droppedDQMEventCountsStats);
    
    getDQMEventSizeMQ().getStats(stats.dqmEventSizeStats);
    getServedDQMEventSizeMQ().getStats(stats.servedDQMEventSizeStats);
    getWrittenDQMEventSizeMQ().getStats(stats.writtenDQMEventSizeStats);
    
    getDQMEventBandwidthMQ().getStats(stats.dqmEventBandwidthStats);
    getServedDQMEventBandwidthMQ().getStats(stats.servedDQMEventBandwidthStats);
    getWrittenDQMEventBandwidthMQ().getStats(stats.writtenDQMEventBandwidthStats);
    
    getNumberOfTopLevelFoldersMQ().getStats(stats.numberOfTopLevelFoldersStats);
    getNumberOfUpdatesMQ().getStats(stats.numberOfUpdatesStats);
    getNumberOfWrittenTopLevelFoldersMQ().getStats(stats.numberOfWrittenTopLevelFoldersStats);

    getNumberOfCompleteUpdatesMQ().getStats(stats.numberOfCompleteUpdatesStats);
  }
  
  
  void DQMEventMonitorCollection::do_calculateStatistics()
  {
    droppedDQMEventCounts_.calculateStatistics();
    
    dqmEventSizes_.calculateStatistics();
    servedDQMEventSizes_.calculateStatistics();
    writtenDQMEventSizes_.calculateStatistics();
    
    MonitoredQuantity::Stats stats;
    dqmEventSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      dqmEventBandwidth_.addSample(stats.getLastValueRate());
    }
    dqmEventBandwidth_.calculateStatistics();
    
    servedDQMEventSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      servedDQMEventBandwidth_.addSample(stats.getLastValueRate());
    }
    servedDQMEventBandwidth_.calculateStatistics();
    
    writtenDQMEventSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      writtenDQMEventBandwidth_.addSample(stats.getLastValueRate());
    }
    writtenDQMEventBandwidth_.calculateStatistics();
    
    numberOfTopLevelFolders_.calculateStatistics();
    numberOfUpdates_.calculateStatistics();
    numberOfWrittenTopLevelFolders_.calculateStatistics();

    numberOfCompleteUpdates_.calculateStatistics();
  }
  
  
  void DQMEventMonitorCollection::do_reset()
  {
    droppedDQMEventCounts_.reset();
    
    dqmEventSizes_.reset();
    servedDQMEventSizes_.reset();
    writtenDQMEventSizes_.reset();
    
    dqmEventBandwidth_.reset();
    servedDQMEventBandwidth_.reset();
    writtenDQMEventBandwidth_.reset();
    
    numberOfTopLevelFolders_.reset();
    numberOfUpdates_.reset();
    numberOfWrittenTopLevelFolders_.reset();

    numberOfCompleteUpdates_.reset();
  }
  
  
  void DQMEventMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
  {
    infoSpaceItems.push_back(std::make_pair("dqmFoldersPerEP", &dqmFoldersPerEP_));
    infoSpaceItems.push_back(std::make_pair("processedDQMEvents", &processedDQMEvents_));
    infoSpaceItems.push_back(std::make_pair("droppedDQMEvents", &droppedDQMEvents_));
    infoSpaceItems.push_back(std::make_pair("discardedDQMEvents", &droppedDQMEvents_));
    infoSpaceItems.push_back(std::make_pair("completeDQMUpdates", &completeDQMUpdates_));
  }
  
  
  void DQMEventMonitorCollection::do_updateInfoSpaceItems()
  {
    DQMEventMonitorCollection::DQMEventStats stats;
    getStats(stats);
    
    dqmFoldersPerEP_ = static_cast<xdata::Double>(
      stats.numberOfUpdatesStats.getValueAverage(MonitoredQuantity::RECENT));
    
    processedDQMEvents_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(stats.dqmEventSizeStats.getSampleCount(MonitoredQuantity::FULL)));
    
    droppedDQMEvents_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(stats.droppedDQMEventCountsStats.getValueSum(MonitoredQuantity::FULL)));
    
    completeDQMUpdates_ = static_cast<xdata::Double>(
      stats.numberOfCompleteUpdatesStats.getValueAverage(MonitoredQuantity::RECENT));
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
