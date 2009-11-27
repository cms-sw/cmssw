// $Id: DQMConsumerMonitorCollection.h,v 1.1 2009/08/24 14:31:11 mommsen Exp $
/// @file: DQMConsumerMonitorCollection.h 

#ifndef StorageManager_DQMConsumerMonitorCollection_h
#define StorageManager_DQMConsumerMonitorCollection_h

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities to track event consumer activity.
   *
   * $Author: mommsen $
   * $Revision: 1.1 $
   * $Date: 2009/08/24 14:31:11 $
   */

  class DQMConsumerMonitorCollection: public ConsumerMonitorCollection
  {

  public:

    explicit DQMConsumerMonitorCollection(const utils::duration_t& updateInterval);

  private:

    // Prevent copying:
    DQMConsumerMonitorCollection( const DQMConsumerMonitorCollection& );
    DQMConsumerMonitorCollection& operator = ( const DQMConsumerMonitorCollection& );

    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::UnsignedInteger32 _dqmConsumers;
    
  };

} // namespace stor

#endif // StorageManager_DQMConsumerMonitorCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
