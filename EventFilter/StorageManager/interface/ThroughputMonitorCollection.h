// $Id: ThroughputMonitorCollection.h,v 1.19 2011/03/07 15:31:32 mommsen Exp $
/// @file: ThroughputMonitorCollection.h 

#ifndef EventFilter_StorageManager_ThroughputMonitorCollection_h
#define EventFilter_StorageManager_ThroughputMonitorCollection_h

#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>

#include "toolbox/mem/Pool.h"
#include "xdata/Double.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/DQMEventQueue.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamQueue.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor {

  /**
   * A collection of MonitoredQuantities to track the flow of data
   * through the storage manager.
   *
   * $Author: mommsen $
   * $Revision: 1.19 $
   * $Date: 2011/03/07 15:31:32 $
   */
  
  class ThroughputMonitorCollection : public MonitorCollection
  {
  public:

    explicit ThroughputMonitorCollection
    (
      const utils::Duration_t& updateInterval,
      const unsigned int& throuphputAveragingCycles
    );

    int getBinCount() const {return binCount_;}

    /**
     * Stores the given memory pool pointer if not yet set.
     * If it is already set, the argument is ignored.
     */
    void setMemoryPoolPointer(toolbox::mem::Pool*);

    void setFragmentQueue(FragmentQueuePtr fragmentQueue) {
      fragmentQueue_ = fragmentQueue;
    }

    const MonitoredQuantity& getPoolUsageMQ() const {
      return poolUsageMQ_;
    }
    MonitoredQuantity& getPoolUsageMQ() {
      return poolUsageMQ_;
    }
 
    const MonitoredQuantity& getFragmentQueueEntryCountMQ() const {
      return entriesInFragmentQueueMQ_;
    }
    MonitoredQuantity& getFragmentQueueEntryCountMQ() {
      return entriesInFragmentQueueMQ_;
    }
 
    const MonitoredQuantity& getFragmentQueueMemoryUsedMQ() const {
      return memoryUsedInFragmentQueueMQ_;
    }
    MonitoredQuantity& getFragmentQueueMemoryUsedMQ() {
      return memoryUsedInFragmentQueueMQ_;
    }

    void addPoppedFragmentSample(double dataSize);

    const MonitoredQuantity& getPoppedFragmentSizeMQ() const {
      return poppedFragmentSizeMQ_;
    }
    MonitoredQuantity& getPoppedFragmentSizeMQ() {
      return poppedFragmentSizeMQ_;
    }

    void addFragmentProcessorIdleSample(utils::Duration_t idleTime);

    const MonitoredQuantity& getFragmentProcessorIdleMQ() const {
      return fragmentProcessorIdleTimeMQ_;
    }
    MonitoredQuantity& getFragmentProcessorIdleMQ() {
      return fragmentProcessorIdleTimeMQ_;
    }

    const MonitoredQuantity& getFragmentStoreEntryCountMQ() const {
      return entriesInFragmentStoreMQ_;
    }
    MonitoredQuantity& getFragmentStoreEntryCountMQ() {
      return entriesInFragmentStoreMQ_;
    }

    const MonitoredQuantity& getFragmentStoreMemoryUsedMQ() const {
      return memoryUsedInFragmentStoreMQ_;
    }
    MonitoredQuantity& getFragmentStoreMemoryUsedMQ() {
      return memoryUsedInFragmentStoreMQ_;
    }

    void setStreamQueue(StreamQueuePtr streamQueue) {
      streamQueue_ = streamQueue;
    }

    const MonitoredQuantity& getStreamQueueEntryCountMQ() const {
      return entriesInStreamQueueMQ_;
    }
    MonitoredQuantity& getStreamQueueEntryCountMQ() {
      return entriesInStreamQueueMQ_;
    }

    const MonitoredQuantity& getStreamQueueMemoryUsedMQ() const {
      return memoryUsedInStreamQueueMQ_;
    }
    MonitoredQuantity& getStreamQueueMemoryUsedMQ() {
      return memoryUsedInStreamQueueMQ_;
    }

    void addPoppedEventSample(double dataSize);

    const MonitoredQuantity& getPoppedEventSizeMQ() const {
      return poppedEventSizeMQ_;
    }
    MonitoredQuantity& getPoppedEventSizeMQ() {
      return poppedEventSizeMQ_;
    }

    void addDiskWriterIdleSample(utils::Duration_t idleTime);

    const MonitoredQuantity& getDiskWriterIdleMQ() const {
      return diskWriterIdleTimeMQ_;
    }
    MonitoredQuantity& getDiskWriterIdleMQ() {
      return diskWriterIdleTimeMQ_;
    }

    void addDiskWriteSample(double dataSize);

    const MonitoredQuantity& getDiskWriteMQ() const {
      return diskWriteSizeMQ_;
    }
    MonitoredQuantity& getDiskWriteMQ() {
      return diskWriteSizeMQ_;
    }

    void setDQMEventQueue(DQMEventQueuePtr dqmEventQueue) {
      dqmEventQueue_ = dqmEventQueue;
    }

    const MonitoredQuantity& getDQMEventQueueEntryCountMQ() const {
      return entriesInDQMEventQueueMQ_;
    }
    MonitoredQuantity& getDQMEventQueueEntryCountMQ() {
      return entriesInDQMEventQueueMQ_;
    }

    const MonitoredQuantity& getDQMEventQueueMemoryUsedMQ() const {
      return memoryUsedInDQMEventQueueMQ_;
    }
    MonitoredQuantity& getDQMEventQueueMemoryUsedMQ() {
      return memoryUsedInDQMEventQueueMQ_;
    }

    void addPoppedDQMEventSample(double dataSize);

    const MonitoredQuantity& getPoppedDQMEventSizeMQ() const {
      return poppedDQMEventSizeMQ_;
    }
    MonitoredQuantity& getPoppedDQMEventSizeMQ() {
      return poppedDQMEventSizeMQ_;
    }

    void addDQMEventProcessorIdleSample(utils::Duration_t idleTime);

    const MonitoredQuantity& getDQMEventProcessorIdleMQ() const {
      return dqmEventProcessorIdleTimeMQ_;
    }
    MonitoredQuantity& getDQMEventProcessorIdleMQ() {
      return dqmEventProcessorIdleTimeMQ_;
    }

    /**
     * Sets the current number of events in the fragment store.
     */
    inline void setFragmentStoreSize(unsigned int size) {
      currentFragmentStoreSize_ = size;
    }

    /**
     * Sets the current number of events in the fragment store.
     */
    inline void setFragmentStoreMemoryUsed(size_t memoryUsed) {
      currentFragmentStoreMemoryUsedMB_ = static_cast<double>(memoryUsed) / (1024*1024);
    }

    struct Stats
    {

      struct Snapshot
      {
        utils::Duration_t duration;
        utils::TimePoint_t absoluteTime;
        double poolUsage; //bytes
        double entriesInFragmentQueue;
        double memoryUsedInFragmentQueue; //MB
        double fragmentQueueRate; //Hz
        double fragmentQueueBandwidth; //MB/s
        double fragmentStoreSize;
        double fragmentStoreMemoryUsed; //MB
        double entriesInStreamQueue;
        double memoryUsedInStreamQueue; //MB
        double streamQueueRate; //Hz
        double streamQueueBandwidth; //MB/s
        double writtenEventsRate; //Hz
        double writtenEventsBandwidth; //MB/s
        double entriesInDQMQueue;
        double memoryUsedInDQMQueue; //MB
        double dqmQueueRate; //Hz
        double dqmQueueBandwidth; //MB/s
        
        double fragmentProcessorBusy; //%
        double diskWriterBusy; //%
        double dqmEventProcessorBusy; //%

        Snapshot();
        Snapshot operator=(const Snapshot&);
        Snapshot operator+=(const Snapshot&);
        Snapshot operator/=(const double&);

      };
      
      typedef std::vector<Snapshot> Snapshots;
      Snapshots snapshots; // time sorted with newest entry first
      Snapshot average;

      void reset();
    };

    /**
     * Write all our collected statistics into the given Stats struct.
     */
    void getStats(Stats&) const;

    /**
     * Write only the sampleCount most recent snapshots into the given Stats struct.
     */
    void getStats(Stats&, const unsigned int sampleCount) const;


  private:

    //Prevent copying of the ThroughputMonitorCollection
    ThroughputMonitorCollection(ThroughputMonitorCollection const&);
    ThroughputMonitorCollection& operator=(ThroughputMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void do_getStats(Stats&, const unsigned int sampleCount) const;

    /**
     * Smooth out binned idle times for the throughput display.
     * Returns the index to be used for the next section to smooth.
     * Note that this method works on the idleTimes and durations
     * lists in *reverse* order.  So, the initial indices should be
     * idleTimes.size()-1.
     */
    void smoothIdleTimes(MonitoredQuantity::Stats&) const;

    int smoothIdleTimesHelper
    (
      std::vector<double>& idleTimes,
      std::vector<utils::Duration_t>& durations,
      int firstIndex, int lastIndex
    ) const;

    void getRateAndBandwidth
    (
      MonitoredQuantity::Stats& stats,
      const int& idx,
      double& rate,
      double& bandwidth
    ) const;

    double calcBusyPercentage(
      MonitoredQuantity::Stats&,
      const int& idx
    ) const;

    void calcPoolUsage();

    const unsigned int binCount_;
    mutable boost::mutex statsMutex_;

    MonitoredQuantity poolUsageMQ_;
    MonitoredQuantity entriesInFragmentQueueMQ_;
    MonitoredQuantity memoryUsedInFragmentQueueMQ_;
    MonitoredQuantity poppedFragmentSizeMQ_;
    MonitoredQuantity fragmentProcessorIdleTimeMQ_;
    MonitoredQuantity entriesInFragmentStoreMQ_;
    MonitoredQuantity memoryUsedInFragmentStoreMQ_;

    MonitoredQuantity entriesInStreamQueueMQ_;
    MonitoredQuantity memoryUsedInStreamQueueMQ_;
    MonitoredQuantity poppedEventSizeMQ_;
    MonitoredQuantity diskWriterIdleTimeMQ_;
    MonitoredQuantity diskWriteSizeMQ_;

    MonitoredQuantity entriesInDQMEventQueueMQ_;
    MonitoredQuantity memoryUsedInDQMEventQueueMQ_;
    MonitoredQuantity poppedDQMEventSizeMQ_;
    MonitoredQuantity dqmEventProcessorIdleTimeMQ_;

    FragmentQueuePtr fragmentQueue_;
    StreamQueuePtr streamQueue_;
    DQMEventQueuePtr dqmEventQueue_;

    unsigned int currentFragmentStoreSize_;
    double currentFragmentStoreMemoryUsedMB_;
    unsigned int throuphputAveragingCycles_;

    toolbox::mem::Pool* pool_;

    xdata::UnsignedInteger32 poolUsage_;                 //I2O message pool usage in bytes
    xdata::UnsignedInteger32 entriesInFragmentQueue_;    //Instantaneous number of fragments in fragment queue
    xdata::Double            memoryUsedInFragmentQueue_; //Instantaneous memory usage of events in fragment queue (MB)
    xdata::Double            fragmentQueueRate_;         //Rate of fragments popped from fragment queue
    xdata::Double            fragmentQueueBandwidth_;    //Bandwidth of fragments popped from fragment queue (MB/s)
    xdata::UnsignedInteger32 fragmentStoreSize_;         //Instantaneous number of fragments in fragment store
    xdata::Double            fragmentStoreMemoryUsed_;   //Instantaneous memory usage of events in fragment store (MB)
    xdata::UnsignedInteger32 entriesInStreamQueue_;      //Instantaneous number of events in stream queue
    xdata::Double            memoryUsedInStreamQueue_;   //Instantaneous memory usage of events in stream queue (MB)
    xdata::Double            streamQueueRate_;           //Rate of events popped from fragment queue
    xdata::Double            streamQueueBandwidth_;      //Bandwidth of events popped from fragment queue (MB/s)
    xdata::Double            writtenEventsRate_;         //Rate of (non-unique) events written to disk
    xdata::Double            writtenEventsBandwidth_;    //Bandwidth of (non-unique) events written to disk
    xdata::UnsignedInteger32 entriesInDQMQueue_;         //Instantaneous number of events in dqm event queue
    xdata::Double            memoryUsedInDQMQueue_;      //Instantaneous memory usage of events in dqm event queue (MB)
    xdata::Double            dqmQueueRate_;              //Rate of events popped from dqm event queue
    xdata::Double            dqmQueueBandwidth_;         //Bandwidth of events popped from dqm event queue (MB/s)

    xdata::Double            fragmentProcessorBusy_;     //Fragment processor busy percentage
    xdata::Double            diskWriterBusy_;            //Disk writer busy percentage
    xdata::Double            dqmEventProcessorBusy_;     //DQM event processor busy percentage
    xdata::Double            averagingTime_;                    //Time in s over which above values are averaged
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_ThroughputMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
