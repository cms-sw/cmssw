// $Id: StreamQueue.h,v 1.3 2009/07/20 13:06:11 mommsen Exp $
/// @file: StreamQueue.h 

#ifndef EventFilter_StorageManager_StreamQueue_h
#define EventFilter_StorageManager_StreamQueue_h

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of events to be written to disk
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/20 13:06:11 $
   */

  typedef ConcurrentQueue<I2OChain> StreamQueue;  
  
} // namespace stor

#endif // StorageManager_StreamQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
