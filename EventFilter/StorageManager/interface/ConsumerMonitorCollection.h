// $Id: ConsumerMonitorCollection.h,v 1.8 2009/08/28 16:41:49 mommsen Exp $
/// @file: ConsumerMonitorCollection.h 

#ifndef StorageManager_ConsumerMonitorCollection_h
#define StorageManager_ConsumerMonitorCollection_h

#include "EventFilter/StorageManager/interface/MonitorCollection.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include <map>

namespace stor {

  class QueueID;
  class MonitoredQuantity;


  /**
   * A collection of MonitoredQuantities to track consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.8 $
   * $Date: 2009/08/28 16:41:49 $
   */

  class ConsumerMonitorCollection: public MonitorCollection
  {

  public:

    explicit ConsumerMonitorCollection(const utils::duration_t& updateInterval);

    /**
       Add queued sample
    */
    void addQueuedEventSample( const QueueID&, const unsigned int& data_size );

    /**
       Add served sample
    */
    void addServedEventSample( const QueueID&, const unsigned int& data_size );

    /**
       Get queued data size. Return false if consumer ID not found.
    */
    bool getQueued( const QueueID& qid, MonitoredQuantity::Stats& result );

    /**
       Get served data size. Return false if consumer ID not found.
    */
    bool getServed( const QueueID& qid, MonitoredQuantity::Stats& result );

    /**
       Reset sizes to zero leaving consumers in
    */
    void resetCounters();

  private:

    // Prevent copying:
    ConsumerMonitorCollection( const ConsumerMonitorCollection& );
    ConsumerMonitorCollection& operator = ( const ConsumerMonitorCollection& );

    typedef std::map< QueueID, boost::shared_ptr<MonitoredQuantity> > ConsStatMap;

    void addEventSampleToMap( const QueueID&, const unsigned int& data_size, ConsStatMap& );
    bool getValueFromMap( const QueueID&, MonitoredQuantity::Stats&, const ConsStatMap& );

    virtual void do_calculateStatistics();
    virtual void do_reset();

    const utils::duration_t _updateInterval;

  protected:

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
