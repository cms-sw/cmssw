// $Id: DQMEventMonitorCollection.cc,v 1.7 2009/09/17 14:43:30 mommsen Exp $
/// @file: DQMEventMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"

using namespace stor;

DQMEventMonitorCollection::DQMEventMonitorCollection(const utils::duration_t& updateInterval) :
MonitorCollection(updateInterval),
_discardedDQMEventCounts(updateInterval, 300),
_dqmEventSizes(updateInterval, 300),
_servedDQMEventSizes(updateInterval, 300),
_writtenDQMEventSizes(updateInterval, 300),
_dqmEventBandwidth(updateInterval, 300),
_servedDQMEventBandwidth(updateInterval, 300),
_writtenDQMEventBandwidth(updateInterval, 300),
_numberOfGroups(updateInterval, 300),
_numberOfUpdates(updateInterval, 300),
_numberOfWrittenGroups(updateInterval, 300)
{}


void DQMEventMonitorCollection::getStats(DQMEventStats& stats) const
{
  getDiscardedDQMEventCountsMQ().getStats(stats.discardedDQMEventCountsStats);

  getDQMEventSizeMQ().getStats(stats.dqmEventSizeStats);
  getServedDQMEventSizeMQ().getStats(stats.servedDQMEventSizeStats);
  getWrittenDQMEventSizeMQ().getStats(stats.writtenDQMEventSizeStats);

  getDQMEventBandwidthMQ().getStats(stats.dqmEventBandwidthStats);
  getServedDQMEventBandwidthMQ().getStats(stats.servedDQMEventBandwidthStats);
  getWrittenDQMEventBandwidthMQ().getStats(stats.writtenDQMEventBandwidthStats);

  getNumberOfGroupsMQ().getStats(stats.numberOfGroupsStats);
  getNumberOfUpdatesMQ().getStats(stats.numberOfUpdatesStats);
  getNumberOfWrittenGroupsMQ().getStats(stats.numberOfWrittenGroupsStats);
}


void DQMEventMonitorCollection::do_calculateStatistics()
{
  _discardedDQMEventCounts.calculateStatistics();

  _dqmEventSizes.calculateStatistics();
  _servedDQMEventSizes.calculateStatistics();
  _writtenDQMEventSizes.calculateStatistics();

  MonitoredQuantity::Stats stats;
  _dqmEventSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _dqmEventBandwidth.addSample(stats.getLastValueRate());
  }
  _dqmEventBandwidth.calculateStatistics();

  _servedDQMEventSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _servedDQMEventBandwidth.addSample(stats.getLastValueRate());
  }
  _servedDQMEventBandwidth.calculateStatistics();

  _writtenDQMEventSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _writtenDQMEventBandwidth.addSample(stats.getLastValueRate());
  }
  _writtenDQMEventBandwidth.calculateStatistics();

  _numberOfGroups.calculateStatistics();
  _numberOfUpdates.calculateStatistics();
  _numberOfWrittenGroups.calculateStatistics();
}


void DQMEventMonitorCollection::do_reset()
{
  _discardedDQMEventCounts.reset();
  
  _dqmEventSizes.reset();
  _servedDQMEventSizes.reset();
  _writtenDQMEventSizes.reset();

  _dqmEventBandwidth.reset();
  _servedDQMEventBandwidth.reset();
  _writtenDQMEventBandwidth.reset();

  _numberOfGroups.reset();
  _numberOfUpdates.reset();
  _numberOfWrittenGroups.reset();
}


void DQMEventMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("dqmFoldersPerEP", &_dqmFoldersPerEP));
  infoSpaceItems.push_back(std::make_pair("processedDQMEvents", &_processedDQMEvents));
  infoSpaceItems.push_back(std::make_pair("discardedDQMEvents", &_discardedDQMEvents));
}


void DQMEventMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats stats;
  getNumberOfUpdatesMQ().getStats(stats);
  _dqmFoldersPerEP = static_cast<xdata::Double>(stats.getValueAverage(MonitoredQuantity::RECENT));
  _processedDQMEvents = static_cast<xdata::UnsignedInteger32>(
    static_cast<unsigned int>(stats.getSampleCount(MonitoredQuantity::FULL)));
  _discardedDQMEvents = static_cast<xdata::UnsignedInteger32>(
    static_cast<unsigned int>(stats.getValueSum(MonitoredQuantity::FULL)));
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
