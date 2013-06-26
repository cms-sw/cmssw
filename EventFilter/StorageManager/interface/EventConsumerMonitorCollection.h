// $Id: EventConsumerMonitorCollection.h,v 1.2 2011/03/07 15:31:31 mommsen Exp $
/// @file: EventConsumerMonitorCollection.h 

#ifndef EventFilter_StorageManager_EventConsumerMonitorCollection_h
#define EventFilter_StorageManager_EventConsumerMonitorCollection_h

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities to track event consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class EventConsumerMonitorCollection: public ConsumerMonitorCollection
  {

  public:

    explicit EventConsumerMonitorCollection(const utils::Duration_t& updateInterval);

  private:

    // Prevent copying:
    EventConsumerMonitorCollection( const EventConsumerMonitorCollection& );
    EventConsumerMonitorCollection& operator = ( const EventConsumerMonitorCollection& );

    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::UnsignedInteger32 eventConsumers_;
    
  };

} // namespace stor

#endif // EventFilter_StorageManager_EventConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
