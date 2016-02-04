// $Id: EventQueueCollection.h,v 1.3 2009/07/20 13:06:10 mommsen Exp $
/// @file: EventQueueCollection.h 

#ifndef StorageManager_EventQueueCollection_h
#define StorageManager_EventQueueCollection_h

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueCollection.h"

namespace stor {

  /**
   * A collection of ConcurrentQueue<I2OChain>.
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/20 13:06:10 $
   */

  typedef QueueCollection<I2OChain> EventQueueCollection;
  
} // namespace stor

#endif // StorageManager_EventQueueCollection_h 



// emacs configuration
// Local Variables: -
// mode: c++ -
// c-basic-offset: 2 -
// indent-tabs-mode: nil -
// End: -
