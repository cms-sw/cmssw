// $Id: ThroughputMonitorCollection.cc,v 1.24 2011/03/07 15:31:32 mommsen Exp $
/// @file: ThroughputMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;

ThroughputMonitorCollection::ThroughputMonitorCollection
(
  const utils::Duration_t& updateInterval,
  const unsigned int& throuphputAveragingCycles
) :
  MonitorCollection(updateInterval),
  binCount_(300),
  poolUsageMQ_(updateInterval, updateInterval*binCount_),
  entriesInFragmentQueueMQ_(updateInterval, updateInterval*binCount_),
  memoryUsedInFragmentQueueMQ_(updateInterval, updateInterval*binCount_),
  poppedFragmentSizeMQ_(updateInterval, updateInterval*binCount_),
  fragmentProcessorIdleTimeMQ_(updateInterval, updateInterval*binCount_),
  entriesInFragmentStoreMQ_(updateInterval, updateInterval*binCount_),
  memoryUsedInFragmentStoreMQ_(updateInterval, updateInterval*binCount_),
  entriesInStreamQueueMQ_(updateInterval, updateInterval*binCount_),
  memoryUsedInStreamQueueMQ_(updateInterval, updateInterval*binCount_),
  poppedEventSizeMQ_(updateInterval, updateInterval*binCount_),
  diskWriterIdleTimeMQ_(updateInterval, updateInterval*binCount_),
  diskWriteSizeMQ_(updateInterval, updateInterval*binCount_),
  entriesInDQMEventQueueMQ_(updateInterval, updateInterval*binCount_),
  memoryUsedInDQMEventQueueMQ_(updateInterval, updateInterval*binCount_),
  poppedDQMEventSizeMQ_(updateInterval, updateInterval*binCount_),
  dqmEventProcessorIdleTimeMQ_(updateInterval, updateInterval*binCount_),
  currentFragmentStoreSize_(0),
  currentFragmentStoreMemoryUsedMB_(0),
  throuphputAveragingCycles_(throuphputAveragingCycles),
  pool_(0)
{}


void ThroughputMonitorCollection::setMemoryPoolPointer(toolbox::mem::Pool* pool)
{
  if ( ! pool_)
    pool_ = pool;
}


void ThroughputMonitorCollection::addPoppedFragmentSample(double dataSize)
{
  poppedFragmentSizeMQ_.addSample(dataSize);
}


void ThroughputMonitorCollection::
addFragmentProcessorIdleSample(utils::Duration_t idleTime)
{
  fragmentProcessorIdleTimeMQ_.addSample(utils::durationToSeconds(idleTime));
}


void ThroughputMonitorCollection::addPoppedEventSample(double dataSize)
{
  poppedEventSizeMQ_.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDiskWriterIdleSample(utils::Duration_t idleTime)
{
  diskWriterIdleTimeMQ_.addSample(utils::durationToSeconds(idleTime));
}


void ThroughputMonitorCollection::addDiskWriteSample(double dataSize)
{
  diskWriteSizeMQ_.addSample(dataSize);
}


void ThroughputMonitorCollection::addPoppedDQMEventSample(double dataSize)
{
  poppedDQMEventSizeMQ_.addSample(dataSize);
}


void ThroughputMonitorCollection::
addDQMEventProcessorIdleSample(utils::Duration_t idleTime)
{
  dqmEventProcessorIdleTimeMQ_.addSample(utils::durationToSeconds(idleTime));
}


void ThroughputMonitorCollection::calcPoolUsage()
{
  if (pool_)
  {
    try {
      pool_->lock();
      poolUsageMQ_.addSample(pool_->getMemoryUsage().getUsed());
      pool_->unlock();
    }
    catch (...)
    {
      pool_->unlock();
    }
  }
  poolUsageMQ_.calculateStatistics();
}


void ThroughputMonitorCollection::getStats(Stats& stats) const
{
  boost::mutex::scoped_lock sl(statsMutex_);
  do_getStats(stats, binCount_);
}


void ThroughputMonitorCollection::getStats(Stats& stats, const unsigned int sampleCount) const
{
  boost::mutex::scoped_lock sl(statsMutex_);
  do_getStats(stats, sampleCount);
}


void ThroughputMonitorCollection::do_getStats(Stats& stats, const unsigned int sampleCount) const
{
  MonitoredQuantity::Stats fqEntryCountMQ, fqMemoryUsedMQ, fragSizeMQ;
  MonitoredQuantity::Stats fpIdleMQ, fsEntryCountMQ, fsMemoryUsedMQ;
  MonitoredQuantity::Stats sqEntryCountMQ, sqMemoryUsedMQ, eventSizeMQ, dwIdleMQ, diskWriteMQ;
  MonitoredQuantity::Stats dqEntryCountMQ, dqMemoryUsedMQ, dqmEventSizeMQ, dqmIdleMQ, poolUsageMQ;
  poolUsageMQ_.getStats(poolUsageMQ);
  entriesInFragmentQueueMQ_.getStats(fqEntryCountMQ);
  memoryUsedInFragmentQueueMQ_.getStats(fqMemoryUsedMQ);
  poppedFragmentSizeMQ_.getStats(fragSizeMQ);
  fragmentProcessorIdleTimeMQ_.getStats(fpIdleMQ);
  entriesInFragmentStoreMQ_.getStats(fsEntryCountMQ);
  memoryUsedInFragmentStoreMQ_.getStats(fsMemoryUsedMQ);
  entriesInStreamQueueMQ_.getStats(sqEntryCountMQ);
  memoryUsedInStreamQueueMQ_.getStats(sqMemoryUsedMQ);
  poppedEventSizeMQ_.getStats(eventSizeMQ);
  diskWriterIdleTimeMQ_.getStats(dwIdleMQ);
  diskWriteSizeMQ_.getStats(diskWriteMQ);
  entriesInDQMEventQueueMQ_.getStats(dqEntryCountMQ);
  memoryUsedInDQMEventQueueMQ_.getStats(dqMemoryUsedMQ);
  poppedDQMEventSizeMQ_.getStats(dqmEventSizeMQ);
  dqmEventProcessorIdleTimeMQ_.getStats(dqmIdleMQ);

  stats.reset();

  smoothIdleTimes(fpIdleMQ);
  smoothIdleTimes(dwIdleMQ);
  smoothIdleTimes(dqmIdleMQ);

  utils::Duration_t relativeTime = fqEntryCountMQ.recentDuration;
  const int lowestBin = sampleCount<binCount_ ? binCount_-sampleCount : 0;
  for (int idx = (binCount_ - 1); idx >= lowestBin; --idx)
  {
    utils::Duration_t binDuration = fqEntryCountMQ.recentBinnedDurations[idx];
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
  int index = binCount_ - 1;
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
  std::vector<utils::Duration_t>& durations,
  int firstIndex, int lastIndex
) const
{
  int workingSize = lastIndex - firstIndex + 1;
  double idleTimeSum = 0;
  double durationSum = 0;

  for (int idx = firstIndex; idx <= lastIndex; ++idx)
  {
    idleTimeSum += idleTimes[idx];
    durationSum += utils::durationToSeconds(durations[idx]);
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
        durations[idx] = utils::secondsToDuration(durationSum / workingSize);
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
  const double recentBinnedDuration = utils::durationToSeconds(stats.recentBinnedDurations[idx]);
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
  else if (stats.recentBinnedValueSums[idx] <= utils::durationToSeconds(stats.recentBinnedDurations[idx]))
  {
    // the thread was busy while it was not idle during the whole reporting duration
    busyPercentage = 100.0 * (1.0 - (stats.recentBinnedValueSums[idx] /
        utils::durationToSeconds(stats.recentBinnedDurations[idx])));
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

  if (fragmentQueue_.get() != 0) {
    entriesInFragmentQueueMQ_.addSample(fragmentQueue_->size());
    memoryUsedInFragmentQueueMQ_.addSample( static_cast<double>(fragmentQueue_->used()) / (1024*1024) );
  }
  if (streamQueue_.get() != 0) {
    entriesInStreamQueueMQ_.addSample(streamQueue_->size());
    memoryUsedInStreamQueueMQ_.addSample( static_cast<double>(streamQueue_->used()) / (1024*1024) );
  }
  if (dqmEventQueue_.get() != 0) {
    entriesInDQMEventQueueMQ_.addSample(dqmEventQueue_->size());
    memoryUsedInDQMEventQueueMQ_.addSample( static_cast<double>(dqmEventQueue_->used()) / (1024*1024) );
  }
  entriesInFragmentStoreMQ_.addSample(currentFragmentStoreSize_);
  memoryUsedInFragmentStoreMQ_.addSample(currentFragmentStoreMemoryUsedMB_);

  entriesInFragmentQueueMQ_.calculateStatistics();
  memoryUsedInFragmentQueueMQ_.calculateStatistics();
  poppedFragmentSizeMQ_.calculateStatistics();
  fragmentProcessorIdleTimeMQ_.calculateStatistics();
  entriesInFragmentStoreMQ_.calculateStatistics();
  memoryUsedInFragmentStoreMQ_.calculateStatistics();
  entriesInStreamQueueMQ_.calculateStatistics();
  memoryUsedInStreamQueueMQ_.calculateStatistics();
  poppedEventSizeMQ_.calculateStatistics();
  diskWriterIdleTimeMQ_.calculateStatistics();
  diskWriteSizeMQ_.calculateStatistics();
  entriesInDQMEventQueueMQ_.calculateStatistics();
  memoryUsedInDQMEventQueueMQ_.calculateStatistics();
  poppedDQMEventSizeMQ_.calculateStatistics();
  dqmEventProcessorIdleTimeMQ_.calculateStatistics();
}


void ThroughputMonitorCollection::do_reset()
{
  poolUsageMQ_.reset();
  entriesInFragmentQueueMQ_.reset();
  memoryUsedInFragmentQueueMQ_.reset();
  poppedFragmentSizeMQ_.reset();
  fragmentProcessorIdleTimeMQ_.reset();
  entriesInFragmentStoreMQ_.reset();
  memoryUsedInFragmentStoreMQ_.reset();
  entriesInStreamQueueMQ_.reset();
  memoryUsedInStreamQueueMQ_.reset();
  poppedEventSizeMQ_.reset();
  diskWriterIdleTimeMQ_.reset();
  diskWriteSizeMQ_.reset();
  entriesInDQMEventQueueMQ_.reset();
  memoryUsedInDQMEventQueueMQ_.reset();
  poppedDQMEventSizeMQ_.reset();
  dqmEventProcessorIdleTimeMQ_.reset();
}


void ThroughputMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("poolUsage", &poolUsage_));
  infoSpaceItems.push_back(std::make_pair("entriesInFragmentQueue", &entriesInFragmentQueue_));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInFragmentQueue", &memoryUsedInFragmentQueue_));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueRate", &fragmentQueueRate_));
  infoSpaceItems.push_back(std::make_pair("fragmentQueueBandwidth", &fragmentQueueBandwidth_));
  infoSpaceItems.push_back(std::make_pair("fragmentStoreSize", &fragmentStoreSize_));
  infoSpaceItems.push_back(std::make_pair("fragmentStoreMemoryUsed", &fragmentStoreMemoryUsed_));
  infoSpaceItems.push_back(std::make_pair("entriesInStreamQueue", &entriesInStreamQueue_));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInStreamQueue", &memoryUsedInStreamQueue_));
  infoSpaceItems.push_back(std::make_pair("streamQueueRate", &streamQueueRate_));
  infoSpaceItems.push_back(std::make_pair("streamQueueBandwidth", &streamQueueBandwidth_));
  infoSpaceItems.push_back(std::make_pair("writtenEventsRate", &writtenEventsRate_));
  infoSpaceItems.push_back(std::make_pair("writtenEventsBandwidth", &writtenEventsBandwidth_));
  infoSpaceItems.push_back(std::make_pair("entriesInDQMQueue", &entriesInDQMQueue_));
  infoSpaceItems.push_back(std::make_pair("memoryUsedInDQMQueue", &memoryUsedInDQMQueue_));
  infoSpaceItems.push_back(std::make_pair("dqmQueueRate", &dqmQueueRate_));
  infoSpaceItems.push_back(std::make_pair("dqmQueueBandwidth", &dqmQueueBandwidth_));
  infoSpaceItems.push_back(std::make_pair("fragmentProcessorBusy", &fragmentProcessorBusy_));
  infoSpaceItems.push_back(std::make_pair("diskWriterBusy", &diskWriterBusy_));
  infoSpaceItems.push_back(std::make_pair("dqmEventProcessorBusy", &dqmEventProcessorBusy_));
  infoSpaceItems.push_back(std::make_pair("averagingTime", &averagingTime_));
}


void ThroughputMonitorCollection::do_updateInfoSpaceItems()
{
  Stats stats;
  getStats(stats, throuphputAveragingCycles_);

  poolUsage_ = static_cast<unsigned int>(stats.average.poolUsage);
  entriesInFragmentQueue_ = static_cast<unsigned int>(stats.average.entriesInFragmentQueue);
  memoryUsedInFragmentQueue_ = stats.average.memoryUsedInFragmentQueue;
  fragmentQueueRate_ = stats.average.fragmentQueueRate;
  fragmentQueueBandwidth_ = stats.average.fragmentQueueBandwidth;
  fragmentStoreSize_ = static_cast<unsigned int>(stats.average.fragmentStoreSize);
  fragmentStoreMemoryUsed_ = stats.average.fragmentStoreMemoryUsed;
  entriesInStreamQueue_ = static_cast<unsigned int>(stats.average.entriesInStreamQueue);
  memoryUsedInStreamQueue_ = stats.average.memoryUsedInStreamQueue;
  streamQueueRate_ = stats.average.streamQueueRate;
  streamQueueBandwidth_ = stats.average.streamQueueBandwidth;
  writtenEventsRate_ = stats.average.writtenEventsRate;
  writtenEventsBandwidth_ = stats.average.writtenEventsBandwidth;
  entriesInDQMQueue_ = static_cast<unsigned int>(stats.average.entriesInDQMQueue);
  memoryUsedInDQMQueue_ = stats.average.memoryUsedInDQMQueue;
  dqmQueueRate_ = stats.average.dqmQueueRate;
  dqmQueueBandwidth_ = stats.average.dqmQueueBandwidth;
  fragmentProcessorBusy_ = stats.average.fragmentProcessorBusy;
  diskWriterBusy_ = stats.average.diskWriterBusy;
  dqmEventProcessorBusy_ = stats.average.dqmEventProcessorBusy;
  averagingTime_ = utils::durationToSeconds(stats.average.duration);
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
