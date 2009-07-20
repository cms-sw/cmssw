// $Id: DQMEventQueue.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DQMEventQueue.h 

#ifndef StorageManager_DQMEventQueue_h
#define StorageManager_DQMEventQueue_h

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of complete DQM events (histograms)
   * waiting to be processed by the DQMEventProcessor
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */

  typedef ConcurrentQueue< I2OChain, RejectNewest<I2OChain> > DQMEventQueue;  
  
} // namespace stor

#endif // StorageManager_DQMEventQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
