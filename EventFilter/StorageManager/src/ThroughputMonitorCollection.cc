// $Id: ThroughputMonitorCollection.cc,v 1.3 2009/06/19 13:10:40 mommsen Exp $

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"

using namespace stor;

ThroughputMonitorCollection::ThroughputMonitorCollection() :
  MonitorCollection(),
  _binCount(300)
{
  _entriesInFragmentQueue.setNewTimeWindowForRecentResults(_binCount);
  _poppedFragmentSize.setNewTimeWindowForRecentResults(_binCount);
  _fragmentProcessorIdleTime.setNewTimeWindowForRecentResults(_binCount);
  _entriesInStreamQueue.setNewTimeWindowForRecentResults(_binCount);
  _poppedEventSize.setNewTimeWindowForRecentResults(_binCount);
  _diskWriterIdleTime.setNewTimeWindowForRecentResults(_binCount);
  _diskWriteSize.setNewTimeWindowForRecentResults(_binCount);
  _entriesInDQMEventQueue.setNewTimeWindowForRecentResults(_binCount);
  _poppedDQMEventSize.setNewTimeWindowForRecentResults(_binCount);
  _dqmEventProcessorIdleTime.setNewTimeWindowForRecentResults(_binCount);
}


void ThroughputMonitorCollection::addPoppedFragmentSample(double dataSize)
{
  _poppedFragmentSize.addSample(dataSize);
}


void ThroughputMonitorCollection::
addFragmentProcessorIdleSample(utils::duration_t idleTime)
{
  _fragmentProcessorIdleTime.addSample(idleTime);
}


void ThroughputMonitorCollection::addPoppedEventSample(double dataSize)
{
  _poppedEventSize.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDiskWriterIdleSample(utils::duration_t idleTime)
{
  _diskWriterIdleTime.addSample(idleTime);
}


void ThroughputMonitorCollection::addDiskWriteSample(double dataSize)
{
  _diskWriteSize.addSample(dataSize);
}


void ThroughputMonitorCollection::addPoppedDQMEventSample(double dataSize)
{
  _poppedDQMEventSize.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDQMEventProcessorIdleSample(utils::duration_t idleTime)
{
  _dqmEventProcessorIdleTime.addSample(idleTime);
}


void ThroughputMonitorCollection::do_calculateStatistics()
{
  if (_fragmentQueue.get() != 0) {
    _entriesInFragmentQueue.addSample(_fragmentQueue->size());
  }
  if (_streamQueue.get() != 0) {
    _entriesInStreamQueue.addSample(_streamQueue->size());
  }
  if (_dqmEventQueue.get() != 0) {
    _entriesInDQMEventQueue.addSample(_dqmEventQueue->size());
  }

  _entriesInFragmentQueue.calculateStatistics();
  _poppedFragmentSize.calculateStatistics();
  _fragmentProcessorIdleTime.calculateStatistics();
  _entriesInStreamQueue.calculateStatistics();
  _poppedEventSize.calculateStatistics();
  _diskWriterIdleTime.calculateStatistics();
  _diskWriteSize.calculateStatistics();
  _entriesInDQMEventQueue.calculateStatistics();
  _poppedDQMEventSize.calculateStatistics();
  _dqmEventProcessorIdleTime.calculateStatistics();
}


void ThroughputMonitorCollection::do_reset()
{
  _entriesInFragmentQueue.reset();
  _poppedFragmentSize.reset();
  _fragmentProcessorIdleTime.reset();
  _entriesInStreamQueue.reset();
  _poppedEventSize.reset();
  _diskWriterIdleTime.reset();
  _diskWriteSize.reset();
  _entriesInDQMEventQueue.reset();
  _poppedDQMEventSize.reset();
  _dqmEventProcessorIdleTime.reset();
}




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
