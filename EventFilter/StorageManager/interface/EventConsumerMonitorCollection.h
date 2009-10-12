// $Id: EventConsumerMonitorCollection.h,v 1.5 2009/08/18 08:54:13 mommsen Exp $
/// @file: EventConsumerMonitorCollection.h 

#ifndef StorageManager_EventConsumerMonitorCollection_h
#define StorageManager_EventConsumerMonitorCollection_h

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities to track event consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2009/08/18 08:54:13 $
   */

  class EventConsumerMonitorCollection: public ConsumerMonitorCollection
  {

  public:

    explicit EventConsumerMonitorCollection(const utils::duration_t& updateInterval);

  private:

    // Prevent copying:
    EventConsumerMonitorCollection( const EventConsumerMonitorCollection& );
    EventConsumerMonitorCollection& operator = ( const EventConsumerMonitorCollection& );

    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::UnsignedInteger32 _eventConsumers;
    
  };

} // namespace stor

#endif // StorageManager_EventConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
