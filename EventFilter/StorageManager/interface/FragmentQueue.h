// $Id: FragmentQueue.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $
/// @file: FragmentQueue.h 

#ifndef EventFilter_StorageManager_FragmentQueue_h
#define EventFilter_StorageManager_FragmentQueue_h

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of event fragments 
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */

  typedef ConcurrentQueue<I2OChain> FragmentQueue;  
  
} // namespace stor

#endif // StorageManager_FragmentQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
