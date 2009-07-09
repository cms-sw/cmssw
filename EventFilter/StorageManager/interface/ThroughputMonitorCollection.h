// $Id: ThroughputMonitorCollection.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $

#ifndef StorageManager_ThroughputMonitorCollection_h
#define StorageManager_ThroughputMonitorCollection_h

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/DQMEventQueue.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamQueue.h"

namespace stor {

  /**
   * A collection of MonitoredQuantities to track the flow of data
   * through the storage manager.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:24 $
   */
  
  class ThroughputMonitorCollection : public MonitorCollection
  {
  public:

    ThroughputMonitorCollection();

    int getBinCount() const {return _binCount;}

    void setFragmentQueue(boost::shared_ptr<FragmentQueue> fragmentQueue) {
      _fragmentQueue = fragmentQueue;
    }

    const MonitoredQuantity& getFragmentQueueEntryCountMQ() const {
      return _entriesInFragmentQueue;
    }
    MonitoredQuantity& getFragmentQueueEntryCountMQ() {
      return _entriesInFragmentQueue;
    }

    void addPoppedFragmentSample(double dataSize);

    const MonitoredQuantity& getPoppedFragmentSizeMQ() const {
      return _poppedFragmentSize;
    }
    MonitoredQuantity& getPoppedFragmentSizeMQ() {
      return _poppedFragmentSize;
    }

    void addFragmentProcessorIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getFragmentProcessorIdleMQ() const {
      return _fragmentProcessorIdleTime;
    }
    MonitoredQuantity& getFragmentProcessorIdleMQ() {
      return _fragmentProcessorIdleTime;
    }

    void setStreamQueue(boost::shared_ptr<StreamQueue> streamQueue) {
      _streamQueue = streamQueue;
    }

    const MonitoredQuantity& getStreamQueueEntryCountMQ() const {
      return _entriesInStreamQueue;
    }
    MonitoredQuantity& getStreamQueueEntryCountMQ() {
      return _entriesInStreamQueue;
    }

    void addPoppedEventSample(double dataSize);

    const MonitoredQuantity& getPoppedEventSizeMQ() const {
      return _poppedEventSize;
    }
    MonitoredQuantity& getPoppedEventSizeMQ() {
      return _poppedEventSize;
    }

    void addDiskWriterIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getDiskWriterIdleMQ() const {
      return _diskWriterIdleTime;
    }
    MonitoredQuantity& getDiskWriterIdleMQ() {
      return _diskWriterIdleTime;
    }

    void addDiskWriteSample(double dataSize);

    const MonitoredQuantity& getDiskWriteMQ() const {
      return _diskWriteSize;
    }
    MonitoredQuantity& getDiskWriteMQ() {
      return _diskWriteSize;
    }

    void setDQMEventQueue(boost::shared_ptr<DQMEventQueue> dqmEventQueue) {
      _dqmEventQueue = dqmEventQueue;
    }

    const MonitoredQuantity& getDQMEventQueueEntryCountMQ() const {
      return _entriesInDQMEventQueue;
    }
    MonitoredQuantity& getDQMEventQueueEntryCountMQ() {
      return _entriesInDQMEventQueue;
    }

    void addPoppedDQMEventSample(double dataSize);

    const MonitoredQuantity& getPoppedDQMEventSizeMQ() const {
      return _poppedDQMEventSize;
    }
    MonitoredQuantity& getPoppedDQMEventSizeMQ() {
      return _poppedDQMEventSize;
    }

    void addDQMEventProcessorIdleSample(utils::duration_t idleTime);

    const MonitoredQuantity& getDQMEventProcessorIdleMQ() const {
      return _dqmEventProcessorIdleTime;
    }
    MonitoredQuantity& getDQMEventProcessorIdleMQ() {
      return _dqmEventProcessorIdleTime;
    }

  private:

    //Prevent copying of the ThroughputMonitorCollection
    ThroughputMonitorCollection(ThroughputMonitorCollection const&);
    ThroughputMonitorCollection& operator=(ThroughputMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();

    int _binCount;

    MonitoredQuantity _entriesInFragmentQueue;
    MonitoredQuantity _poppedFragmentSize;
    MonitoredQuantity _fragmentProcessorIdleTime;

    MonitoredQuantity _entriesInStreamQueue;
    MonitoredQuantity _poppedEventSize;
    MonitoredQuantity _diskWriterIdleTime;
    MonitoredQuantity _diskWriteSize;

    MonitoredQuantity _entriesInDQMEventQueue;
    MonitoredQuantity _poppedDQMEventSize;
    MonitoredQuantity _dqmEventProcessorIdleTime;

    boost::shared_ptr<FragmentQueue> _fragmentQueue;
    boost::shared_ptr<StreamQueue> _streamQueue;
    boost::shared_ptr<DQMEventQueue> _dqmEventQueue;

  };
  
} // namespace stor

#endif // StorageManager_ThroughputMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
