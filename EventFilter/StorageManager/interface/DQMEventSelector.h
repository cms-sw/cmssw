// $Id: DQMEventSelector.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DQMEventSelector.h 

#ifndef StorageManager_DQMEventSelector_h
#define StorageManager_DQMEventSelector_h

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/EventSelector.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"

namespace stor
{
  /**
   * DQM event selector
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */

  class DQMEventSelector
  {

  public:
    
    DQMEventSelector( const DQMEventConsumerRegistrationInfo* ri ):
    _topLevelFolderName( ri->topLevelFolderName() ),
    _queueId( ri->queueId() ),
    _stale( false )
    {};
    
    /**
     * Returns true if the DQM event stored in the I2OChain
     * passes this event selection.
     */
    bool acceptEvent( const I2OChain& );
    
    /**
     * Returns the ID of the queue corresponding to this selector.
     */
    const QueueID& queueId() const { return _queueId; }
    
    /**
       Check if stale:
    */
    bool isStale() const { return _stale; }

    /**
       Mark as stale:
    */
    void markAsStale() { _stale = true; }

    /**
       Mark as active:
    */
    void markAsActive() { _stale = false; }

  private:

    std::string _topLevelFolderName;
    QueueID _queueId;
    bool _stale;

  };

}

#endif // StorageManager_DQMEventSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
