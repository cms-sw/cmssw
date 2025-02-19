// $Id: ConsumerMonitorCollection.h,v 1.12 2011/03/07 15:31:31 mommsen Exp $
/// @file: ConsumerMonitorCollection.h 

#ifndef EventFilter_StorageManager_ConsumerMonitorCollection_h
#define EventFilter_StorageManager_ConsumerMonitorCollection_h

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include <map>

namespace stor {

  class MonitoredQuantity;


  /**
   * A collection of MonitoredQuantities to track consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.12 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class ConsumerMonitorCollection: public MonitorCollection
  {

  public:

    struct TotalStats
    {
      MonitoredQuantity::Stats queuedStats;
      MonitoredQuantity::Stats droppedStats;
      MonitoredQuantity::Stats servedStats;
    };

    explicit ConsumerMonitorCollection
    (
      const utils::Duration_t& updateInterval,
      const utils::Duration_t& recentDuration
    );

    /**
       Add queued sample
    */
    void addQueuedEventSample( const QueueID&, const unsigned int& data_size );

    /**
       Add number of dropped events
    */
    void addDroppedEvents( const QueueID&, const size_t& count );

    /**
       Add served sample
    */
    void addServedEventSample( const QueueID&, const unsigned int& data_size );

    /**
       Get queued data size. Return false if consumer ID not found.
    */
    bool getQueued( const QueueID& qid, MonitoredQuantity::Stats& result ) const;

    /**
       Get served data size. Return false if consumer ID not found.
    */
    bool getServed( const QueueID& qid, MonitoredQuantity::Stats& result ) const;

    /**
       Get number of dropped events. Return false if consumer ID not found.
    */
    bool getDropped( const QueueID& qid, MonitoredQuantity::Stats& result ) const;

    /**
       Get the summary statistics for all consumers
    */
    void getTotalStats( TotalStats& ) const;

    /**
       Reset sizes to zero leaving consumers in
    */
    void resetCounters();

  private:

    // Prevent copying:
    ConsumerMonitorCollection( const ConsumerMonitorCollection& );
    ConsumerMonitorCollection& operator = ( const ConsumerMonitorCollection& );

    typedef std::map<QueueID, MonitoredQuantityPtr> ConsStatMap;

    void addEventSampleToMap( const QueueID&, const unsigned int& data_size, ConsStatMap& );
    bool getValueFromMap( const QueueID&, MonitoredQuantity::Stats&, const ConsStatMap& ) const;

    virtual void do_calculateStatistics();
    virtual void do_reset();

    const utils::Duration_t updateInterval_;
    const utils::Duration_t recentDuration_;
    MonitoredQuantity totalQueuedMQ_;
    MonitoredQuantity totalDroppedMQ_;
    MonitoredQuantity totalServedMQ_;

  protected:

    ConsStatMap qmap_; // queued
    ConsStatMap dmap_; // dropped
    ConsStatMap smap_; // served

    mutable boost::mutex mutex_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_ConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
