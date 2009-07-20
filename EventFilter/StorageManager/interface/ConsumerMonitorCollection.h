// -*- c++ -*-
// $Id: ConsumerMonitorCollection.h,v 1.3 2009/07/09 15:34:44 mommsen Exp $
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
   * $Revision: 1.3 $
   * $Date: 2009/07/09 15:34:44 $
   */

  class ConsumerMonitorCollection: public MonitorCollection
  {

  public:

    ConsumerMonitorCollection();

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

  };

} // namespace stor

#endif // StorageManager_ConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
