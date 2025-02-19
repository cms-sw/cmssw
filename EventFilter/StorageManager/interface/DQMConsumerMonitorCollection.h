// $Id: DQMConsumerMonitorCollection.h,v 1.2 2011/03/07 15:31:31 mommsen Exp $
/// @file: DQMConsumerMonitorCollection.h 

#ifndef EventFilter_StorageManager_DQMConsumerMonitorCollection_h
#define EventFilter_StorageManager_DQMConsumerMonitorCollection_h

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

  class DQMConsumerMonitorCollection: public ConsumerMonitorCollection
  {

  public:

    explicit DQMConsumerMonitorCollection(const utils::Duration_t& updateInterval);

  private:

    // Prevent copying:
    DQMConsumerMonitorCollection( const DQMConsumerMonitorCollection& );
    DQMConsumerMonitorCollection& operator = ( const DQMConsumerMonitorCollection& );

    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::UnsignedInteger32 dqmConsumers_;
    
  };

} // namespace stor

#endif // EventFilter_StorageManager_DQMConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
