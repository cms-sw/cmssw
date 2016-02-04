// $Id: DQMEventQueue.h,v 1.4 2010/02/18 11:20:26 mommsen Exp $
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
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2010/02/18 11:20:26 $
   */

  typedef ConcurrentQueue< I2OChain, KeepNewest<I2OChain> > DQMEventQueue;  
  
} // namespace stor

#endif // StorageManager_DQMEventQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
