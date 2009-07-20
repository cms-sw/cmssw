// $Id: EventQueueCollection.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $
/// @file: EventQueueCollection.h 

#ifndef StorageManager_EventQueueCollection_h
#define StorageManager_EventQueueCollection_h

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueCollection.h"

namespace stor {

  /**
   * A collection of ConcurrentQueue<I2OChain>.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:22 $
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
