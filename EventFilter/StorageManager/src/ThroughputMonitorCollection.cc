// $Id: ThroughputMonitorCollection.cc,v 1.8 2009/08/18 08:55:12 mommsen Exp $
/// @file: ThroughputMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"

using namespace stor;

ThroughputMonitorCollection::ThroughputMonitorCollection(const utils::duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  _binCount(static_cast<int>(300/updateInterval)),
  _poolUsage(updateInterval, _binCount),
  _entriesInFragmentQueue(updateInterval, _binCount),
  _poppedFragmentSize(updateInterval, _binCount),
  _fragmentProcessorIdleTime(updateInterval, _binCount),
  _entriesInFragmentStore(updateInterval, _binCount),
  _entriesInStreamQueue(updateInterval, _binCount),
  _poppedEventSize(updateInterval, _binCount),
  _diskWriterIdleTime(updateInterval, _binCount),
  _diskWriteSize(updateInterval, _binCount),
  _entriesInDQMEventQueue(updateInterval, _binCount),
  _poppedDQMEventSize(updateInterval, _binCount),
  _dqmEventProcessorIdleTime(updateInterval, _binCount),
  _currentFragmentStoreSize(0),
  _pool(0)
{}


void ThroughputMonitorCollection::setMemoryPoolPointer(toolbox::mem::Pool* pool)
{
  if ( ! _pool)
    _pool = pool;
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


void ThroughputMonitorCollection::calcPoolUsage()
{
  if (_pool)
  {
    try {
      _pool->lock();
      _poolUsage.addSample( _pool->getMemoryUsage().getUsed() );
      _pool->unlock();
    }
    catch (...)
    {
      _pool->unlock();
    }
  }
  _poolUsage.calculateStatistics();
}


void ThroughputMonitorCollection::do_calculateStatistics()
{
  calcPoolUsage();

  if (_fragmentQueue.get() != 0) {
    _entriesInFragmentQueue.addSample(_fragmentQueue->size());
  }
  if (_streamQueue.get() != 0) {
    _entriesInStreamQueue.addSample(_streamQueue->size());
  }
  if (_dqmEventQueue.get() != 0) {
    _entriesInDQMEventQueue.addSample(_dqmEventQueue->size());
  }
  _entriesInFragmentStore.addSample(getFragmentStoreSize());

  _entriesInFragmentQueue.calculateStatistics();
  _poppedFragmentSize.calculateStatistics();
  _fragmentProcessorIdleTime.calculateStatistics();
  _entriesInFragmentStore.calculateStatistics();
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
  _poolUsage.reset();
  _entriesInFragmentQueue.reset();
  _poppedFragmentSize.reset();
  _fragmentProcessorIdleTime.reset();
  _entriesInFragmentStore.reset();
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
