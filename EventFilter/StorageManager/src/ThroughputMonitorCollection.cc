// $Id: ThroughputMonitorCollection.cc,v 1.10 2009/08/24 16:39:26 mommsen Exp $
/// @file: ThroughputMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"

using namespace stor;

ThroughputMonitorCollection::ThroughputMonitorCollection(const utils::duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  _binCount(static_cast<int>(300/updateInterval)),
  _poolUsageMQ(updateInterval, _binCount),
  _entriesInFragmentQueueMQ(updateInterval, _binCount),
  _poppedFragmentSizeMQ(updateInterval, _binCount),
  _fragmentProcessorIdleTimeMQ(updateInterval, _binCount),
  _entriesInFragmentStoreMQ(updateInterval, _binCount),
  _entriesInStreamQueueMQ(updateInterval, _binCount),
  _poppedEventSizeMQ(updateInterval, _binCount),
  _diskWriterIdleTimeMQ(updateInterval, _binCount),
  _diskWriteSizeMQ(updateInterval, _binCount),
  _entriesInDQMEventQueueMQ(updateInterval, _binCount),
  _poppedDQMEventSizeMQ(updateInterval, _binCount),
  _dqmEventProcessorIdleTimeMQ(updateInterval, _binCount),
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
  _poppedFragmentSizeMQ.addSample(dataSize);
}


void ThroughputMonitorCollection::
addFragmentProcessorIdleSample(utils::duration_t idleTime)
{
  _fragmentProcessorIdleTimeMQ.addSample(idleTime);
}


void ThroughputMonitorCollection::addPoppedEventSample(double dataSize)
{
  _poppedEventSizeMQ.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDiskWriterIdleSample(utils::duration_t idleTime)
{
  _diskWriterIdleTimeMQ.addSample(idleTime);
}


void ThroughputMonitorCollection::addDiskWriteSample(double dataSize)
{
  _diskWriteSizeMQ.addSample(dataSize);
}


void ThroughputMonitorCollection::addPoppedDQMEventSample(double dataSize)
{
  _poppedDQMEventSizeMQ.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDQMEventProcessorIdleSample(utils::duration_t idleTime)
{
  _dqmEventProcessorIdleTimeMQ.addSample(idleTime);
}


void ThroughputMonitorCollection::calcPoolUsage()
{
  if (_pool)
  {
    try {
      _pool->lock();
      _poolUsageMQ.addSample( _pool->getMemoryUsage().getUsed() );
      _pool->unlock();
    }
    catch (...)
    {
      _pool->unlock();
    }
  }
  _poolUsageMQ.calculateStatistics();
}


void ThroughputMonitorCollection::do_calculateStatistics()
{
  calcPoolUsage();

  if (_fragmentQueue.get() != 0) {
    _entriesInFragmentQueueMQ.addSample(_fragmentQueue->size());
  }
  if (_streamQueue.get() != 0) {
    _entriesInStreamQueueMQ.addSample(_streamQueue->size());
  }
  if (_dqmEventQueue.get() != 0) {
    _entriesInDQMEventQueueMQ.addSample(_dqmEventQueue->size());
  }
  _entriesInFragmentStoreMQ.addSample(getFragmentStoreSize());

  _entriesInFragmentQueueMQ.calculateStatistics();
  _poppedFragmentSizeMQ.calculateStatistics();
  _fragmentProcessorIdleTimeMQ.calculateStatistics();
  _entriesInFragmentStoreMQ.calculateStatistics();
  _entriesInStreamQueueMQ.calculateStatistics();
  _poppedEventSizeMQ.calculateStatistics();
  _diskWriterIdleTimeMQ.calculateStatistics();
  _diskWriteSizeMQ.calculateStatistics();
  _entriesInDQMEventQueueMQ.calculateStatistics();
  _poppedDQMEventSizeMQ.calculateStatistics();
  _dqmEventProcessorIdleTimeMQ.calculateStatistics();
}


void ThroughputMonitorCollection::do_reset()
{
  _poolUsageMQ.reset();
  _entriesInFragmentQueueMQ.reset();
  _poppedFragmentSizeMQ.reset();
  _fragmentProcessorIdleTimeMQ.reset();
  _entriesInFragmentStoreMQ.reset();
  _entriesInStreamQueueMQ.reset();
  _poppedEventSizeMQ.reset();
  _diskWriterIdleTimeMQ.reset();
  _diskWriteSizeMQ.reset();
  _entriesInDQMEventQueueMQ.reset();
  _poppedDQMEventSizeMQ.reset();
  _dqmEventProcessorIdleTimeMQ.reset();
}


void ThroughputMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("poolUsage", &_poolUsage));
  infoSpaceItems.push_back(std::make_pair("entriesInFragmentQueue", &_entriesInFragmentQueue));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueRate", &_fragmentQueueRate));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueBandwidth", &_fragmentQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("fragmentStoreSize", &_fragmentStoreSize));
  infoSpaceItems.push_back(std::make_pair("entriesInStreamQueue", &_entriesInStreamQueue));
  infoSpaceItems.push_back(std::make_pair("streamQueueRate", &_streamQueueRate));
  infoSpaceItems.push_back(std::make_pair("streamQueueBandwidth", &_streamQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("diskWriterBusy", &_diskWriterBusy));
  infoSpaceItems.push_back(std::make_pair("dqmEventProcessorBusy", &_dqmEventProcessorBusy));
  infoSpaceItems.push_back(std::make_pair("entriesInDQMQueue", &_entriesInDQMQueue));
  infoSpaceItems.push_back(std::make_pair("dqmQueueRate", &_dqmQueueRate));
  infoSpaceItems.push_back(std::make_pair("dqmQueueBandwidth", &_dqmQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("writtenEventsRate", &_writtenEventsRate));
  infoSpaceItems.push_back(std::make_pair("writtenEventsBandwidth", &_writtenEventsBandwidth));
  infoSpaceItems.push_back(std::make_pair("fragmentProcessorBusy", &_fragmentProcessorBusy));
}


void ThroughputMonitorCollection::do_updateInfoSpaceItems()
{
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
