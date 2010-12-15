// $Id: ThroughputMonitorCollection.cc,v 1.22 2010/12/14 12:56:52 mommsen Exp $
/// @file: ThroughputMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;

ThroughputMonitorCollection::ThroughputMonitorCollection
(
  const utils::duration_t& updateInterval,
  const unsigned int& throuphputAveragingCycles
) :
  MonitorCollection(updateInterval),
  _binCount(300),
  _poolUsageMQ(updateInterval, updateInterval*_binCount),
  _entriesInFragmentQueueMQ(updateInterval, updateInterval*_binCount),
  _memoryUsedInFragmentQueueMQ(updateInterval, updateInterval*_binCount),
  _poppedFragmentSizeMQ(updateInterval, updateInterval*_binCount),
  _fragmentProcessorIdleTimeMQ(updateInterval, updateInterval*_binCount),
  _entriesInFragmentStoreMQ(updateInterval, updateInterval*_binCount),
  _memoryUsedInFragmentStoreMQ(updateInterval, updateInterval*_binCount),
  _entriesInStreamQueueMQ(updateInterval, updateInterval*_binCount),
  _memoryUsedInStreamQueueMQ(updateInterval, updateInterval*_binCount),
  _poppedEventSizeMQ(updateInterval, updateInterval*_binCount),
  _diskWriterIdleTimeMQ(updateInterval, updateInterval*_binCount),
  _diskWriteSizeMQ(updateInterval, updateInterval*_binCount),
  _entriesInDQMEventQueueMQ(updateInterval, updateInterval*_binCount),
  _memoryUsedInDQMEventQueueMQ(updateInterval, updateInterval*_binCount),
  _poppedDQMEventSizeMQ(updateInterval, updateInterval*_binCount),
  _dqmEventProcessorIdleTimeMQ(updateInterval, updateInterval*_binCount),
  _currentFragmentStoreSize(0),
  _currentFragmentStoreMemoryUsedMB(0),
  _throuphputAveragingCycles(throuphputAveragingCycles),
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
  _fragmentProcessorIdleTimeMQ.addSample(utils::duration_to_seconds(idleTime));
}


void ThroughputMonitorCollection::addPoppedEventSample(double dataSize)
{
  _poppedEventSizeMQ.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDiskWriterIdleSample(utils::duration_t idleTime)
{
  _diskWriterIdleTimeMQ.addSample(utils::duration_to_seconds(idleTime));
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
  _dqmEventProcessorIdleTimeMQ.addSample(utils::duration_to_seconds(idleTime));
}


void ThroughputMonitorCollection::calcPoolUsage()
{
  if (_pool)
  {
    try {
      _pool->lock();
      _poolUsageMQ.addSample(_pool->getMemoryUsage().getUsed());
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
  MonitoredQuantity::Stats fqEntryCountMQ, fqMemoryUsedMQ, fragSizeMQ;
  MonitoredQuantity::Stats fpIdleMQ, fsEntryCountMQ, fsMemoryUsedMQ;
  MonitoredQuantity::Stats sqEntryCountMQ, sqMemoryUsedMQ, eventSizeMQ, dwIdleMQ, diskWriteMQ;
  MonitoredQuantity::Stats dqEntryCountMQ, dqMemoryUsedMQ, dqmEventSizeMQ, dqmIdleMQ, poolUsageMQ;
  _poolUsageMQ.getStats(poolUsageMQ);
  _entriesInFragmentQueueMQ.getStats(fqEntryCountMQ);
  _memoryUsedInFragmentQueueMQ.getStats(fqMemoryUsedMQ);
  _poppedFragmentSizeMQ.getStats(fragSizeMQ);
  _fragmentProcessorIdleTimeMQ.getStats(fpIdleMQ);
  _entriesInFragmentStoreMQ.getStats(fsEntryCountMQ);
  _memoryUsedInFragmentStoreMQ.getStats(fsMemoryUsedMQ);
  _entriesInStreamQueueMQ.getStats(sqEntryCountMQ);
  _memoryUsedInStreamQueueMQ.getStats(sqMemoryUsedMQ);
  _poppedEventSizeMQ.getStats(eventSizeMQ);
  _diskWriterIdleTimeMQ.getStats(dwIdleMQ);
  _diskWriteSizeMQ.getStats(diskWriteMQ);
  _entriesInDQMEventQueueMQ.getStats(dqEntryCountMQ);
  _memoryUsedInDQMEventQueueMQ.getStats(dqMemoryUsedMQ);
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
    if (binDuration < boost::posix_time::milliseconds(10)) continue; //avoid very short durations

    Stats::Snapshot snapshot;

    snapshot.duration = binDuration;
    snapshot.absoluteTime = fqEntryCountMQ.recentBinnedSnapshotTimes[idx];

    // memory pool usage
    snapshot.poolUsage = poolUsageMQ.recentBinnedSampleCounts[idx]>0 ? 
      poolUsageMQ.recentBinnedValueSums[idx]/poolUsageMQ.recentBinnedSampleCounts[idx] :
      0;

    // number of fragments in fragment queue
    snapshot.entriesInFragmentQueue = fqEntryCountMQ.recentBinnedSampleCounts[idx]>0 ?
      fqEntryCountMQ.recentBinnedValueSums[idx]/fqEntryCountMQ.recentBinnedSampleCounts[idx] :
      0;

    // memory usage in fragment queue
    snapshot.memoryUsedInFragmentQueue = fqMemoryUsedMQ.recentBinnedSampleCounts[idx]>0 ?
      fqMemoryUsedMQ.recentBinnedValueSums[idx]/fqMemoryUsedMQ.recentBinnedSampleCounts[idx] :
      0;

    // rate/bandwidth of fragments popped from fragment queue
    getRateAndBandwidth(fragSizeMQ, idx, snapshot.fragmentQueueRate, snapshot.fragmentQueueBandwidth);

    // number of events in fragment store
    snapshot.fragmentStoreSize = fsEntryCountMQ.recentBinnedSampleCounts[idx]>0 ?
      fsEntryCountMQ.recentBinnedValueSums[idx]/fsEntryCountMQ.recentBinnedSampleCounts[idx]>0 :
      0;

    // memory usage in fragment store
    snapshot.fragmentStoreMemoryUsed = fsMemoryUsedMQ.recentBinnedSampleCounts[idx]>0 ?
      fsMemoryUsedMQ.recentBinnedValueSums[idx]/fsMemoryUsedMQ.recentBinnedSampleCounts[idx] :
      0;

    // number of events in stream queue
    snapshot.entriesInStreamQueue = sqEntryCountMQ.recentBinnedSampleCounts[idx]>0 ?
      sqEntryCountMQ.recentBinnedValueSums[idx]/sqEntryCountMQ.recentBinnedSampleCounts[idx]>0 :
      0;

    // memory usage in stream queue
    snapshot.memoryUsedInStreamQueue = sqMemoryUsedMQ.recentBinnedSampleCounts[idx]>0 ?
      sqMemoryUsedMQ.recentBinnedValueSums[idx]/sqMemoryUsedMQ.recentBinnedSampleCounts[idx] :
      0;

    // rate/bandwidth of events popped from stream queue
    getRateAndBandwidth(eventSizeMQ, idx, snapshot.streamQueueRate, snapshot.streamQueueBandwidth);

    // rate/bandwidth of events written to disk
    getRateAndBandwidth(diskWriteMQ, idx, snapshot.writtenEventsRate, snapshot.writtenEventsBandwidth);

    // number of dqm events in DQMEvent queue
    snapshot.entriesInDQMQueue = dqEntryCountMQ.recentBinnedSampleCounts[idx]>0 ?
      dqEntryCountMQ.recentBinnedValueSums[idx]/dqEntryCountMQ.recentBinnedSampleCounts[idx] :
      0;

    // memory usage in DQMEvent queue
    snapshot.memoryUsedInDQMQueue = dqMemoryUsedMQ.recentBinnedSampleCounts[idx]>0 ?
      dqMemoryUsedMQ.recentBinnedValueSums[idx]/dqMemoryUsedMQ.recentBinnedSampleCounts[idx] :
      0;

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

  const size_t snapshotCount = stats.snapshots.size();
  if (snapshotCount > 0)
  {
    stats.average /= snapshotCount;
  }
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
  double idleTimeSum = 0;
  double durationSum = 0;

  for (int idx = firstIndex; idx <= lastIndex; ++idx)
  {
    idleTimeSum += idleTimes[idx];
    durationSum += utils::duration_to_seconds(durations[idx]);
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
        durations[idx] = utils::seconds_to_duration(durationSum / workingSize);
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
  const double recentBinnedDuration = utils::duration_to_seconds(stats.recentBinnedDurations[idx]);
  if (recentBinnedDuration > 0)
  {
    rate =
      stats.recentBinnedSampleCounts[idx] / recentBinnedDuration;
    
    bandwidth =
      stats.recentBinnedValueSums[idx] / (1024*1024) 
      / recentBinnedDuration;
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
  else if (stats.recentBinnedValueSums[idx] <= utils::duration_to_seconds(stats.recentBinnedDurations[idx]))
  {
    // the thread was busy while it was not idle during the whole reporting duration
    busyPercentage = 100.0 * (1.0 - (stats.recentBinnedValueSums[idx] /
        utils::duration_to_seconds(stats.recentBinnedDurations[idx])));
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
    _memoryUsedInFragmentQueueMQ.addSample( static_cast<double>(_fragmentQueue->used()) / (1024*1024) );
  }
  if (_streamQueue.get() != 0) {
    _entriesInStreamQueueMQ.addSample(_streamQueue->size());
    _memoryUsedInStreamQueueMQ.addSample( static_cast<double>(_streamQueue->used()) / (1024*1024) );
  }
  if (_dqmEventQueue.get() != 0) {
    _entriesInDQMEventQueueMQ.addSample(_dqmEventQueue->size());
    _memoryUsedInDQMEventQueueMQ.addSample( static_cast<double>(_dqmEventQueue->used()) / (1024*1024) );
  }
  _entriesInFragmentStoreMQ.addSample(_currentFragmentStoreSize);
  _memoryUsedInFragmentStoreMQ.addSample(_currentFragmentStoreMemoryUsedMB);

  _entriesInFragmentQueueMQ.calculateStatistics();
  _memoryUsedInFragmentQueueMQ.calculateStatistics();
  _poppedFragmentSizeMQ.calculateStatistics();
  _fragmentProcessorIdleTimeMQ.calculateStatistics();
  _entriesInFragmentStoreMQ.calculateStatistics();
  _memoryUsedInFragmentStoreMQ.calculateStatistics();
  _entriesInStreamQueueMQ.calculateStatistics();
  _memoryUsedInStreamQueueMQ.calculateStatistics();
  _poppedEventSizeMQ.calculateStatistics();
  _diskWriterIdleTimeMQ.calculateStatistics();
  _diskWriteSizeMQ.calculateStatistics();
  _entriesInDQMEventQueueMQ.calculateStatistics();
  _memoryUsedInDQMEventQueueMQ.calculateStatistics();
  _poppedDQMEventSizeMQ.calculateStatistics();
  _dqmEventProcessorIdleTimeMQ.calculateStatistics();
}


void ThroughputMonitorCollection::do_reset()
{
  _poolUsageMQ.reset();
  _entriesInFragmentQueueMQ.reset();
  _memoryUsedInFragmentQueueMQ.reset();
  _poppedFragmentSizeMQ.reset();
  _fragmentProcessorIdleTimeMQ.reset();
  _entriesInFragmentStoreMQ.reset();
  _memoryUsedInFragmentStoreMQ.reset();
  _entriesInStreamQueueMQ.reset();
  _memoryUsedInStreamQueueMQ.reset();
  _poppedEventSizeMQ.reset();
  _diskWriterIdleTimeMQ.reset();
  _diskWriteSizeMQ.reset();
  _entriesInDQMEventQueueMQ.reset();
  _memoryUsedInDQMEventQueueMQ.reset();
  _poppedDQMEventSizeMQ.reset();
  _dqmEventProcessorIdleTimeMQ.reset();
}


void ThroughputMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("poolUsage", &_poolUsage));
  infoSpaceItems.push_back(std::make_pair("entriesInFragmentQueue", &_entriesInFragmentQueue));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInFragmentQueue", &_memoryUsedInFragmentQueue));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueRate", &_fragmentQueueRate));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueBandwidth", &_fragmentQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("fragmentStoreSize", &_fragmentStoreSize));
  infoSpaceItems.push_back(std::make_pair("fragmentStoreMemoryUsed", &_fragmentStoreMemoryUsed));
  infoSpaceItems.push_back(std::make_pair("entriesInStreamQueue", &_entriesInStreamQueue));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInStreamQueue", &_memoryUsedInStreamQueue));
  infoSpaceItems.push_back(std::make_pair("streamQueueRate", &_streamQueueRate));
  infoSpaceItems.push_back(std::make_pair("streamQueueBandwidth", &_streamQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("writtenEventsRate", &_writtenEventsRate));
  infoSpaceItems.push_back(std::make_pair("writtenEventsBandwidth", &_writtenEventsBandwidth));
  infoSpaceItems.push_back(std::make_pair("entriesInDQMQueue", &_entriesInDQMQueue));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInDQMQueue", &_memoryUsedInDQMQueue));
  infoSpaceItems.push_back(std::make_pair("dqmQueueRate", &_dqmQueueRate));
  infoSpaceItems.push_back(std::make_pair("dqmQueueBandwidth", &_dqmQueueBandwidth));
  infoSpaceItems.push_back(std::make_pair("fragmentProcessorBusy", &_fragmentProcessorBusy));
  infoSpaceItems.push_back(std::make_pair("diskWriterBusy", &_diskWriterBusy));
  infoSpaceItems.push_back(std::make_pair("dqmEventProcessorBusy", &_dqmEventProcessorBusy));
  infoSpaceItems.push_back(std::make_pair("averagingTime", &_averagingTime));
}


void ThroughputMonitorCollection::do_updateInfoSpaceItems()
{
  Stats stats;
  getStats(stats, _throuphputAveragingCycles);

  _poolUsage = static_cast<unsigned int>(stats.average.poolUsage);
  _entriesInFragmentQueue = static_cast<unsigned int>(stats.average.entriesInFragmentQueue);
  _memoryUsedInFragmentQueue = stats.average.memoryUsedInFragmentQueue;
  _fragmentQueueRate = stats.average.fragmentQueueRate;
  _fragmentQueueBandwidth = stats.average.fragmentQueueBandwidth;
  _fragmentStoreSize = static_cast<unsigned int>(stats.average.fragmentStoreSize);
  _fragmentStoreMemoryUsed = stats.average.fragmentStoreMemoryUsed;
  _entriesInStreamQueue = static_cast<unsigned int>(stats.average.entriesInStreamQueue);
  _memoryUsedInStreamQueue = stats.average.memoryUsedInStreamQueue;
  _streamQueueRate = stats.average.streamQueueRate;
  _streamQueueBandwidth = stats.average.streamQueueBandwidth;
  _writtenEventsRate = stats.average.writtenEventsRate;
  _writtenEventsBandwidth = stats.average.writtenEventsBandwidth;
  _entriesInDQMQueue = static_cast<unsigned int>(stats.average.entriesInDQMQueue);
  _memoryUsedInDQMQueue = stats.average.memoryUsedInDQMQueue;
  _dqmQueueRate = stats.average.dqmQueueRate;
  _dqmQueueBandwidth = stats.average.dqmQueueBandwidth;
  _fragmentProcessorBusy = stats.average.fragmentProcessorBusy;
  _diskWriterBusy = stats.average.diskWriterBusy;
  _dqmEventProcessorBusy = stats.average.dqmEventProcessorBusy;
  _averagingTime = utils::duration_to_seconds(stats.average.duration);
}


ThroughputMonitorCollection::Stats::Snapshot::Snapshot() :
duration(boost::posix_time::seconds(0)),
poolUsage(0),
entriesInFragmentQueue(0),
memoryUsedInFragmentQueue(0),
fragmentQueueRate(0),
fragmentQueueBandwidth(0),
fragmentStoreSize(0),
fragmentStoreMemoryUsed(0),
entriesInStreamQueue(0),
memoryUsedInStreamQueue(0),
streamQueueRate(0),
streamQueueBandwidth(0),
writtenEventsRate(0),
writtenEventsBandwidth(0),
entriesInDQMQueue(0),
memoryUsedInDQMQueue(0),
dqmQueueRate(0),
dqmQueueBandwidth(0),
fragmentProcessorBusy(0),
diskWriterBusy(0),
dqmEventProcessorBusy(0)
{}


ThroughputMonitorCollection::Stats::Snapshot
ThroughputMonitorCollection::Stats::Snapshot::operator=(const Snapshot& other)
{
  duration = other.duration;
  poolUsage = other.poolUsage;
  entriesInFragmentQueue = other.entriesInFragmentQueue;
  memoryUsedInFragmentQueue = other.memoryUsedInFragmentQueue;
  fragmentQueueRate = other.fragmentQueueRate;
  fragmentQueueBandwidth = other.fragmentQueueBandwidth;
  fragmentStoreSize = other.fragmentStoreSize;
  fragmentStoreMemoryUsed = other.fragmentStoreMemoryUsed;
  entriesInStreamQueue = other.entriesInStreamQueue;
  memoryUsedInStreamQueue = other.memoryUsedInStreamQueue;
  streamQueueRate = other.streamQueueRate;
  streamQueueBandwidth = other.streamQueueBandwidth;
  writtenEventsRate = other.writtenEventsRate;
  writtenEventsBandwidth = other.writtenEventsBandwidth;
  entriesInDQMQueue = other.entriesInDQMQueue;
  memoryUsedInDQMQueue = other.memoryUsedInDQMQueue;
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
  duration += other.duration;
  poolUsage += other.poolUsage;
  entriesInFragmentQueue += other.entriesInFragmentQueue;
  memoryUsedInFragmentQueue += other.memoryUsedInFragmentQueue;
  fragmentQueueRate += other.fragmentQueueRate;
  fragmentQueueBandwidth += other.fragmentQueueBandwidth;
  fragmentStoreSize += other.fragmentStoreSize;
  fragmentStoreMemoryUsed += other.fragmentStoreMemoryUsed;
  entriesInStreamQueue += other.entriesInStreamQueue;
  memoryUsedInStreamQueue += other.memoryUsedInStreamQueue;
  streamQueueRate += other.streamQueueRate;
  streamQueueBandwidth += other.streamQueueBandwidth;
  writtenEventsRate += other.writtenEventsRate;
  writtenEventsBandwidth += other.writtenEventsBandwidth;
  entriesInDQMQueue += other.entriesInDQMQueue;
  memoryUsedInDQMQueue += other.memoryUsedInDQMQueue;
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
  poolUsage /= value;
  entriesInFragmentQueue /= value;
  memoryUsedInFragmentQueue /= value;
  fragmentQueueRate /= value;
  fragmentQueueBandwidth /= value;
  fragmentStoreSize /= value;
  fragmentStoreMemoryUsed /= value;
  entriesInStreamQueue /= value;
  memoryUsedInStreamQueue /= value;
  streamQueueRate /= value;
  streamQueueBandwidth /= value;
  writtenEventsRate /= value;
  writtenEventsBandwidth /= value;
  entriesInDQMQueue /= value;
  memoryUsedInDQMQueue /= value;
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
