// $Id: StreamQueue.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $
/// @file: StreamQueue.h 

#ifndef EventFilter_StorageManager_StreamQueue_h
#define EventFilter_StorageManager_StreamQueue_h

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of events to be written to disk
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:24 $
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
