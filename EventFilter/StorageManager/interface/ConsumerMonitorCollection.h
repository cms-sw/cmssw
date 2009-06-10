// -*- c++ -*-
// $Id$

#ifndef CONSUMERMONITORCOLLECTION
#define CONSUMERMONITORCOLLECTION

#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include <map>

namespace stor
{

  class ConsumerMonitorCollection: public MonitorCollection
  {

  public:

    explicit ConsumerMonitorCollection( xdaq::Application* );

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
    virtual void do_updateInfoSpace();

    typedef std::map< QueueID, boost::shared_ptr<MonitoredQuantity> > ConsStatMap;

    ConsStatMap _qmap; // queued
    ConsStatMap _smap; // served

    mutable boost::mutex _mutex;

  };

} // namespace stor

#endif
