// $Id: DQMEventMonitorCollection.cc,v 1.2 2009/06/10 08:15:25 dshpakov Exp $

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"

using namespace stor;

DQMEventMonitorCollection::DQMEventMonitorCollection() :
MonitorCollection()
{
  _dqmEventSizes.setNewTimeWindowForRecentResults(300);
  _servedDQMEventSizes.setNewTimeWindowForRecentResults(300);
  _writtenDQMEventSizes.setNewTimeWindowForRecentResults(300);
  _dqmEventBandwidth.setNewTimeWindowForRecentResults(300);
  _servedDQMEventBandwidth.setNewTimeWindowForRecentResults(300);
  _writtenDQMEventBandwidth.setNewTimeWindowForRecentResults(300);
  _numberOfGroups.setNewTimeWindowForRecentResults(300);
  _numberOfUpdates.setNewTimeWindowForRecentResults(300);
  _numberOfWrittenGroups.setNewTimeWindowForRecentResults(300);
}


void DQMEventMonitorCollection::getStats(DQMEventStats& stats) const
{
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




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
