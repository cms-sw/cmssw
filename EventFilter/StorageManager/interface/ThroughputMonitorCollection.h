// $Id: ThroughputMonitorCollection.h,v 1.14 2010/02/15 15:11:28 mommsen Exp $
/// @file: ThroughputMonitorCollection.h 

#ifndef StorageManager_ThroughputMonitorCollection_h
#define StorageManager_ThroughputMonitorCollection_h

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
   * $Revision: 1.14 $
   * $Date: 2010/02/15 15:11:28 $
   */
  
  class ThroughputMonitorCollection : public MonitorCollection
  {
  public:

    explicit ThroughputMonitorCollection(const utils::duration_t& updateInterval);

    int getBinCount() const {return _binCount;}

    /**
     * Stores the given memory pool pointer if not yet set.
     * If it is already set, the argument is ignored.
     */
    void setMemoryPoolPointer(toolbox::mem::Pool*);

    void setFragmentQueue(boost::shared_ptr<FragmentQueue> fragmentQueue) {
      _fragmentQueue = fragmentQueue;
    }

    const MonitoredQuantity& getPoolUsageMQ() const {
      return _poolUsageMQ;
    }
    MonitoredQuantity& getPoolUsageMQ() {
      return _poolUsageMQ;
    }
 
    const MonitoredQuantity& getFragmentQueueEntryCountMQ() const {
      return _entriesInFragmentQueueMQ;
    }
    MonitoredQuantity& getFragmentQueueEntryCountMQ() {
      return _entriesInFragmentQueueMQ;
    }
 
    const MonitoredQuantity& getFragmentQueueMemoryUsedMQ() const {
      return _memoryUsedInFragmentQueueMQ;
    }
    MonitoredQuantity& getFragmentQueueMemoryUsedMQ() {
      return _memoryUsedInFragmentQueueMQ;
    }

    void addPoppedFragmentSample(double dataSize);

    const MonitoredQuantity& getPoppedFragmentSizeMQ() const {
      return _poppedFragmentSizeMQ;
    }
    MonitoredQuantity& getPoppedFragmentSizeMQ() {
      return _poppedFragmentSizeMQ;
    }

    void addFragmentProcessorIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getFragmentProcessorIdleMQ() const {
      return _fragmentProcessorIdleTimeMQ;
    }
    MonitoredQuantity& getFragmentProcessorIdleMQ() {
      return _fragmentProcessorIdleTimeMQ;
    }

    const MonitoredQuantity& getFragmentStoreEntryCountMQ() const {
      return _entriesInFragmentStoreMQ;
    }
    MonitoredQuantity& getFragmentStoreEntryCountMQ() {
      return _entriesInFragmentStoreMQ;
    }

    const MonitoredQuantity& getFragmentStoreMemoryUsedMQ() const {
      return _memoryUsedInFragmentStoreMQ;
    }
    MonitoredQuantity& getFragmentStoreMemoryUsedMQ() {
      return _memoryUsedInFragmentStoreMQ;
    }

    void setStreamQueue(boost::shared_ptr<StreamQueue> streamQueue) {
      _streamQueue = streamQueue;
    }

    const MonitoredQuantity& getStreamQueueEntryCountMQ() const {
      return _entriesInStreamQueueMQ;
    }
    MonitoredQuantity& getStreamQueueEntryCountMQ() {
      return _entriesInStreamQueueMQ;
    }

    const MonitoredQuantity& getStreamQueueMemoryUsedMQ() const {
      return _memoryUsedInStreamQueueMQ;
    }
    MonitoredQuantity& getStreamQueueMemoryUsedMQ() {
      return _memoryUsedInStreamQueueMQ;
    }

    void addPoppedEventSample(double dataSize);

    const MonitoredQuantity& getPoppedEventSizeMQ() const {
      return _poppedEventSizeMQ;
    }
    MonitoredQuantity& getPoppedEventSizeMQ() {
      return _poppedEventSizeMQ;
    }

    void addDiskWriterIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getDiskWriterIdleMQ() const {
      return _diskWriterIdleTimeMQ;
    }
    MonitoredQuantity& getDiskWriterIdleMQ() {
      return _diskWriterIdleTimeMQ;
    }

    void addDiskWriteSample(double dataSize);

    const MonitoredQuantity& getDiskWriteMQ() const {
      return _diskWriteSizeMQ;
    }
    MonitoredQuantity& getDiskWriteMQ() {
      return _diskWriteSizeMQ;
    }

    void setDQMEventQueue(boost::shared_ptr<DQMEventQueue> dqmEventQueue) {
      _dqmEventQueue = dqmEventQueue;
    }

    const MonitoredQuantity& getDQMEventQueueEntryCountMQ() const {
      return _entriesInDQMEventQueueMQ;
    }
    MonitoredQuantity& getDQMEventQueueEntryCountMQ() {
      return _entriesInDQMEventQueueMQ;
    }

    const MonitoredQuantity& getDQMEventQueueMemoryUsedMQ() const {
      return _memoryUsedInDQMEventQueueMQ;
    }
    MonitoredQuantity& getDQMEventQueueMemoryUsedMQ() {
      return _memoryUsedInDQMEventQueueMQ;
    }

    void addPoppedDQMEventSample(double dataSize);

    const MonitoredQuantity& getPoppedDQMEventSizeMQ() const {
      return _poppedDQMEventSizeMQ;
    }
    MonitoredQuantity& getPoppedDQMEventSizeMQ() {
      return _poppedDQMEventSizeMQ;
    }

    void addDQMEventProcessorIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getDQMEventProcessorIdleMQ() const {
      return _dqmEventProcessorIdleTimeMQ;
    }
    MonitoredQuantity& getDQMEventProcessorIdleMQ() {
      return _dqmEventProcessorIdleTimeMQ;
    }

    /**
     * Sets the current number of events in the fragment store.
     */
    inline void setFragmentStoreSize(unsigned int size) {
      _currentFragmentStoreSize = size;
    }

    /**
     * Sets the current number of events in the fragment store.
     */
    inline void setFragmentStoreMemoryUsed(size_t memoryUsed) {
      _currentFragmentStoreMemoryUsedMB = static_cast<double>(memoryUsed) / (1024*1024);
    }

    struct Stats
    {

      struct Snapshot
      {
        double relativeTime; //s since first entry
        utils::time_point_t absoluteTime;
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
      std::vector<utils::duration_t>& durations,
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

    const unsigned int _binCount;
    mutable boost::mutex _statsMutex;

    MonitoredQuantity _poolUsageMQ;
    MonitoredQuantity _entriesInFragmentQueueMQ;
    MonitoredQuantity _memoryUsedInFragmentQueueMQ;
    MonitoredQuantity _poppedFragmentSizeMQ;
    MonitoredQuantity _fragmentProcessorIdleTimeMQ;
    MonitoredQuantity _entriesInFragmentStoreMQ;
    MonitoredQuantity _memoryUsedInFragmentStoreMQ;

    MonitoredQuantity _entriesInStreamQueueMQ;
    MonitoredQuantity _memoryUsedInStreamQueueMQ;
    MonitoredQuantity _poppedEventSizeMQ;
    MonitoredQuantity _diskWriterIdleTimeMQ;
    MonitoredQuantity _diskWriteSizeMQ;

    MonitoredQuantity _entriesInDQMEventQueueMQ;
    MonitoredQuantity _memoryUsedInDQMEventQueueMQ;
    MonitoredQuantity _poppedDQMEventSizeMQ;
    MonitoredQuantity _dqmEventProcessorIdleTimeMQ;

    boost::shared_ptr<FragmentQueue> _fragmentQueue;
    boost::shared_ptr<StreamQueue> _streamQueue;
    boost::shared_ptr<DQMEventQueue> _dqmEventQueue;

    unsigned int _currentFragmentStoreSize;
    double _currentFragmentStoreMemoryUsedMB;

    toolbox::mem::Pool* _pool;

    xdata::UnsignedInteger32 _poolUsage;                 //I2O message pool usage in bytes
    xdata::UnsignedInteger32 _entriesInFragmentQueue;    //Instantaneous number of fragments in fragment queue
    xdata::Double            _memoryUsedInFragmentQueue; //Instantaneous memory usage of events in fragment queue (MB)
    xdata::Double            _fragmentQueueRate;         //Rate of fragments popped from fragment queue
    xdata::Double            _fragmentQueueBandwidth;    //Bandwidth of fragments popped from fragment queue (MB/s)
    xdata::UnsignedInteger32 _fragmentStoreSize;         //Instantaneous number of fragments in fragment store
    xdata::Double            _fragmentStoreMemoryUsed;   //Instantaneous memory usage of events in fragment store (MB)
    xdata::UnsignedInteger32 _entriesInStreamQueue;      //Instantaneous number of events in stream queue
    xdata::Double            _memoryUsedInStreamQueue;   //Instantaneous memory usage of events in stream queue (MB)
    xdata::Double            _streamQueueRate;           //Rate of events popped from fragment queue
    xdata::Double            _streamQueueBandwidth;      //Bandwidth of events popped from fragment queue (MB/s)
    xdata::Double            _writtenEventsRate;         //Rate of (non-unique) events written to disk
    xdata::Double            _writtenEventsBandwidth;    //Bandwidth of (non-unique) events written to disk
    xdata::UnsignedInteger32 _entriesInDQMQueue;         //Instantaneous number of events in dqm event queue
    xdata::Double            _memoryUsedInDQMQueue;      //Instantaneous memory usage of events in dqm event queue (MB)
    xdata::Double            _dqmQueueRate;              //Rate of events popped from dqm event queue
    xdata::Double            _dqmQueueBandwidth;         //Bandwidth of events popped from dqm event queue (MB/s)

    xdata::Double            _fragmentProcessorBusy;     //Fragment processor busy percentage
    xdata::Double            _diskWriterBusy;            //Disk writer busy percentage
    xdata::Double            _dqmEventProcessorBusy;     //DQM event processor busy percentage
  };
  
} // namespace stor

#endif // StorageManager_ThroughputMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
