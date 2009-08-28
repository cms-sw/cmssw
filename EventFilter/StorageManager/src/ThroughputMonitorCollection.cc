// $Id: ThroughputMonitorCollection.cc,v 1.12 2009/08/27 14:41:31 mommsen Exp $
/// @file: ThroughputMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

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


void ThroughputMonitorCollection::getStats(Stats& stats) const
{
  boost::mutex::scoped_lock sl(_statsMutex);
  do_getStats(stats, _binCount);
}


void ThroughputMonitorCollection::getStats(Stats& stats, const unsigned int sampleCount) const
{
  boost::mutex::scoped_lock sl(_statsMutex);
  do_getStats(stats, sampleCount);
}


void ThroughputMonitorCollection::do_getStats(Stats& stats, const unsigned int sampleCount) const
{
  MonitoredQuantity::Stats fqEntryCountMQ, fragSizeMQ, fpIdleMQ, fsEntryCountMQ;
  MonitoredQuantity::Stats sqEntryCountMQ, eventSizeMQ, dwIdleMQ, diskWriteMQ;
  MonitoredQuantity::Stats dqEntryCountMQ, dqmEventSizeMQ, dqmIdleMQ, poolUsageMQ;
  _poolUsageMQ.getStats(poolUsageMQ);
  _entriesInFragmentQueueMQ.getStats(fqEntryCountMQ);
  _poppedFragmentSizeMQ.getStats(fragSizeMQ);
  _fragmentProcessorIdleTimeMQ.getStats(fpIdleMQ);
  _entriesInFragmentStoreMQ.getStats(fsEntryCountMQ);
  _entriesInStreamQueueMQ.getStats(sqEntryCountMQ);
  _poppedEventSizeMQ.getStats(eventSizeMQ);
  _diskWriterIdleTimeMQ.getStats(dwIdleMQ);
  _diskWriteSizeMQ.getStats(diskWriteMQ);
  _entriesInDQMEventQueueMQ.getStats(dqEntryCountMQ);
  _poppedDQMEventSizeMQ.getStats(dqmEventSizeMQ);
  _dqmEventProcessorIdleTimeMQ.getStats(dqmIdleMQ);

  stats.reset();

  smoothIdleTimes(fpIdleMQ);
  smoothIdleTimes(dwIdleMQ);
  smoothIdleTimes(dqmIdleMQ);

  utils::duration_t relativeTime = fqEntryCountMQ.recentDuration;
  const int lowestBin = sampleCount<_binCount ? _binCount-sampleCount : 0;
  for (int idx = (_binCount - 1); idx >= lowestBin; --idx)
  {
    utils::duration_t binDuration = fqEntryCountMQ.recentBinnedDurations[idx];
    relativeTime -= binDuration;
    if (binDuration < 0.01) continue; //avoid very short durations

    Stats::Snapshot snapshot;

    snapshot.relativeTime = relativeTime;

    // memory pool usage
    snapshot.poolUsage = poolUsageMQ.recentBinnedValueSums[idx];

    // number of fragments in fragment queue
    snapshot.entriesInFragmentQueue = fqEntryCountMQ.recentBinnedValueSums[idx];

    // rate/bandwidth of fragments popped from fragment queue
    getRateAndBandwidth(fragSizeMQ, idx, snapshot.fragmentQueueRate, snapshot.fragmentQueueBandwidth);

    // number of events in fragment store
    snapshot.fragmentStoreSize = fsEntryCountMQ.recentBinnedValueSums[idx];

    // number of events in stream queue
    snapshot.entriesInStreamQueue = sqEntryCountMQ.recentBinnedValueSums[idx];

    // rate/bandwidth of events popped from stream queue
    getRateAndBandwidth(eventSizeMQ, idx, snapshot.streamQueueRate, snapshot.streamQueueBandwidth);

    // rate/bandwidth of events written to disk
    getRateAndBandwidth(diskWriteMQ, idx, snapshot.writtenEventsRate, snapshot.writtenEventsBandwidth);

    // number of dqm events in DQMEvent queue
    snapshot.entriesInDQMQueue = dqEntryCountMQ.recentBinnedValueSums[idx];

    // rate/bandwidth of dqm events popped from DQMEvent queue
    getRateAndBandwidth(dqmEventSizeMQ, idx, snapshot.dqmQueueRate, snapshot.dqmQueueBandwidth);

    // fragment processor thread busy percentage
    snapshot.fragmentProcessorBusy =
      calcBusyPercentage(fpIdleMQ, idx);

    // disk writer thread busy percentage
    snapshot.diskWriterBusy =
      calcBusyPercentage(dwIdleMQ, idx);

    // DQMEvent processor thread busy percentage
    snapshot.dqmEventProcessorBusy =
      calcBusyPercentage(dqmIdleMQ, idx);

    stats.average += snapshot;
    stats.snapshots.push_back(snapshot);
  }

  if (sampleCount > 1) stats.average /= sampleCount;
}


void ThroughputMonitorCollection::smoothIdleTimes(MonitoredQuantity::Stats& stats) const
{
  int index = _binCount - 1;
  while (index >= 0)
  {
    index = smoothIdleTimesHelper(stats.recentBinnedValueSums,
                                  stats.recentBinnedDurations,
                                  index, index);
  }
}


int ThroughputMonitorCollection::smoothIdleTimesHelper
(
  std::vector<double>& idleTimes,
  std::vector<utils::duration_t>& durations,
  int firstIndex, int lastIndex
) const
{
  int workingSize = lastIndex - firstIndex + 1;
  double idleTimeSum = 0.0;
  double durationSum = 0.0;

  for (int idx = firstIndex; idx <= lastIndex; ++idx)
  {
    idleTimeSum += idleTimes[idx];
    durationSum += durations[idx];
  }

  if (idleTimeSum > durationSum && firstIndex > 0)
  {
    return smoothIdleTimesHelper(idleTimes, durations, firstIndex-1, lastIndex);
  }
  else
  {
    if (lastIndex > firstIndex)
    {
      for (int idx = firstIndex; idx <= lastIndex; ++idx)
      {
        idleTimes[idx] = idleTimeSum / workingSize;
        durations[idx] = durationSum / workingSize;
      }
    }
    return (firstIndex - 1);
  }
}


void ThroughputMonitorCollection::getRateAndBandwidth
(
  MonitoredQuantity::Stats& stats,
  const int& idx,
  double& rate,
  double& bandwidth
) const
{
  if (stats.recentBinnedDurations[idx] > 0.0)
    {
      rate =
        stats.recentBinnedSampleCounts[idx] / stats.recentBinnedDurations[idx];

      bandwidth =
        stats.recentBinnedValueSums[idx] / (1024*1024) 
        / stats.recentBinnedDurations[idx];
    }
}


double ThroughputMonitorCollection::calcBusyPercentage
(
  MonitoredQuantity::Stats& stats,
  const int& idx
) const
{
  double busyPercentage;
  if (stats.recentBinnedSampleCounts[idx] == 0)
  {
    // the thread did not log any idle time
    busyPercentage = 100;
  }
  else if (stats.recentBinnedSampleCounts[idx] == 1)
  {
    // only one sample means that we waited a whole second on a queue
    // this should only happen if deq_timed_wait timeout >= statistics calculation period
    busyPercentage = 0;
  }
  else if (stats.recentBinnedValueSums[idx] <= stats.recentBinnedDurations[idx])
  {
    // the thread was busy while it was not idle during the whole reporting duration
    busyPercentage = 100.0 * (1.0 - (stats.recentBinnedValueSums[idx] /
        stats.recentBinnedDurations[idx]));
    busyPercentage += 0.5;
  }
  else
  {
    // the process logged more idle time than the whole reporting duration
    // this can happen due to rounding issues.
    busyPercentage = 0;
  }

  return busyPercentage;
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
  infoSpaceItems.push_back(std::make_pair("writtenEventsRate", &_writtenEventsRate));
  infoSpaceItems.push_back(std::make_pair("writtenEventsBandwidth", &_writtenEventsBandwidth));
  infoSpaceItems.push_back(std::make_pair("entriesInDQMQueue", &_entriesInDQMQueue));
  infoSpaceItems.push_back(std::make_pair("dqmQueueRate", &_dqmQueueRate));
  infoSpaceItems.push_back(std::make_pair("dqmQueueBandwidth", &_dqmQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("fragmentProcessorBusy", &_fragmentProcessorBusy));
  infoSpaceItems.push_back(std::make_pair("diskWriterBusy", &_diskWriterBusy));
  infoSpaceItems.push_back(std::make_pair("dqmEventProcessorBusy", &_dqmEventProcessorBusy));
}


void ThroughputMonitorCollection::do_updateInfoSpaceItems()
{
  Stats stats;
  getStats(stats, 1);
  if ( stats.snapshots.empty() ) return;

  Stats::Snapshots::const_iterator it = stats.snapshots.begin();

  _poolUsage = static_cast<unsigned int>(it->poolUsage);
  _entriesInFragmentQueue = static_cast<unsigned int>(it->entriesInFragmentQueue);
  _fragmentQueueRate = it->fragmentQueueRate;
  _fragmentQueueBandwidth = it->fragmentQueueBandwidth;
  _fragmentStoreSize = static_cast<unsigned int>(it->fragmentStoreSize);
  _entriesInStreamQueue = static_cast<unsigned int>(it->entriesInStreamQueue);
  _streamQueueRate = it->streamQueueRate;
  _streamQueueBandwidth = it->streamQueueBandwidth;
  _writtenEventsRate = it->writtenEventsRate;
  _writtenEventsBandwidth = it->writtenEventsBandwidth;
  _entriesInDQMQueue = static_cast<unsigned int>(it->entriesInDQMQueue);
  _dqmQueueRate = it->dqmQueueRate;
  _dqmQueueBandwidth = it->dqmQueueBandwidth;
  _fragmentProcessorBusy = it->fragmentProcessorBusy;
  _diskWriterBusy = it->diskWriterBusy;
  _dqmEventProcessorBusy = it->dqmEventProcessorBusy;
}


ThroughputMonitorCollection::Stats::Snapshot::Snapshot() :
relativeTime(0),
poolUsage(0),
entriesInFragmentQueue(0),
fragmentQueueRate(0),
fragmentQueueBandwidth(0),
fragmentStoreSize(0),
entriesInStreamQueue(0),
streamQueueRate(0),
streamQueueBandwidth(0),
writtenEventsRate(0),
writtenEventsBandwidth(0),
entriesInDQMQueue(0),
dqmQueueRate(0),
dqmQueueBandwidth(0),
fragmentProcessorBusy(0),
diskWriterBusy(0),
dqmEventProcessorBusy(0)
{}


ThroughputMonitorCollection::Stats::Snapshot
ThroughputMonitorCollection::Stats::Snapshot::operator=(const Snapshot& other)
{
  relativeTime = other.relativeTime;
  poolUsage = other.poolUsage;
  entriesInFragmentQueue = other.entriesInFragmentQueue;
  fragmentQueueRate = other.fragmentQueueRate;
  fragmentQueueBandwidth = other.fragmentQueueBandwidth;
  fragmentStoreSize = other.fragmentStoreSize;
  entriesInStreamQueue = other.entriesInStreamQueue;
  streamQueueRate = other.streamQueueRate;
  streamQueueBandwidth = other.streamQueueBandwidth;
  writtenEventsRate = other.writtenEventsRate;
  writtenEventsBandwidth = other.writtenEventsBandwidth;
  entriesInDQMQueue = other.entriesInDQMQueue;
  dqmQueueRate = other.dqmQueueRate;
  dqmQueueBandwidth = other.dqmQueueBandwidth;
  fragmentProcessorBusy = other.fragmentProcessorBusy;
  diskWriterBusy = other.diskWriterBusy;
  dqmEventProcessorBusy = other.dqmEventProcessorBusy;

  return *this;
}


ThroughputMonitorCollection::Stats::Snapshot
ThroughputMonitorCollection::Stats::Snapshot::operator+=(const Snapshot& other)
{
  relativeTime = -1;
  poolUsage += other.poolUsage;
  entriesInFragmentQueue += other.entriesInFragmentQueue;
  fragmentQueueRate += other.fragmentQueueRate;
  fragmentQueueBandwidth += other.fragmentQueueBandwidth;
  fragmentStoreSize += other.fragmentStoreSize;
  entriesInStreamQueue += other.entriesInStreamQueue;
  streamQueueRate += other.streamQueueRate;
  streamQueueBandwidth += other.streamQueueBandwidth;
  writtenEventsRate += other.writtenEventsRate;
  writtenEventsBandwidth += other.writtenEventsBandwidth;
  entriesInDQMQueue += other.entriesInDQMQueue;
  dqmQueueRate += other.dqmQueueRate;
  dqmQueueBandwidth += other.dqmQueueBandwidth;
  fragmentProcessorBusy += other.fragmentProcessorBusy;
  diskWriterBusy += other.diskWriterBusy;
  dqmEventProcessorBusy += other.dqmEventProcessorBusy;

  return *this;
}


ThroughputMonitorCollection::Stats::Snapshot
ThroughputMonitorCollection::Stats::Snapshot::operator/=(const double& value)
{
  relativeTime = -1;
  poolUsage /= value;
  entriesInFragmentQueue /= value;
  fragmentQueueRate /= value;
  fragmentQueueBandwidth /= value;
  fragmentStoreSize /= value;
  entriesInStreamQueue /= value;
  streamQueueRate /= value;
  streamQueueBandwidth /= value;
  writtenEventsRate /= value;
  writtenEventsBandwidth /= value;
  entriesInDQMQueue /= value;
  dqmQueueRate /= value;
  dqmQueueBandwidth /= value;
  fragmentProcessorBusy /= value;
  diskWriterBusy /= value;
  dqmEventProcessorBusy /= value;

  return *this;
}


void ThroughputMonitorCollection::Stats::reset()
{
  snapshots.clear();
  Snapshot empty;
  average = empty;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
