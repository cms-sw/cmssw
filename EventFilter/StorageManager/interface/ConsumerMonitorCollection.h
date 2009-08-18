// -*- c++ -*-
// $Id: ConsumerMonitorCollection.h,v 1.4 2009/07/20 13:06:10 mommsen Exp $
/// @file: ConsumerMonitorCollection.h 

#ifndef StorageManager_ConsumerMonitorCollection_h
#define StorageManager_ConsumerMonitorCollection_h

#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include <map>

namespace stor {

  /**
   * A collection of MonitoredQuantities to track consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2009/07/20 13:06:10 $
   */

  class ConsumerMonitorCollection: public MonitorCollection
  {

  public:

    explicit ConsumerMonitorCollection(const utils::duration_t& updateInterval);

    /**
       Add queued sample
    */
    void addQueuedEventSample( QueueID qid, unsigned int data_size );

    /**
       Add served sample
    */
    void addServedEventSample( QueueID qid, unsigned int data_size );

    /**
       Get queued data size. Return false if consumer ID not found.
    */
    bool getQueued( QueueID qid, MonitoredQuantity::Stats& result );

    /**
       Get served data size. Return false if consumer ID not found.
    */
    bool getServed( QueueID qid, MonitoredQuantity::Stats& result );

    /**
       Reset sizes to zero leaving consumers in
    */
    void resetCounters();

  private:

    // Prevent copying:
    ConsumerMonitorCollection( const ConsumerMonitorCollection& );
    ConsumerMonitorCollection& operator = ( const ConsumerMonitorCollection& );

    virtual void do_calculateStatistics();
    virtual void do_reset();

    typedef std::map< QueueID, boost::shared_ptr<MonitoredQuantity> > ConsStatMap;

    ConsStatMap _qmap; // queued
    ConsStatMap _smap; // served

    mutable boost::mutex _mutex;

    const utils::duration_t _updateInterval;

  };

} // namespace stor

#endif // StorageManager_ConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
