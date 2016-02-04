// $Id: FragmentQueue.h,v 1.4 2011/03/07 15:31:32 mommsen Exp $
/// @file: FragmentQueue.h 

#ifndef EventFilter_StorageManager_FragmentQueue_h
#define EventFilter_StorageManager_FragmentQueue_h

#include "boost/shared_ptr.hpp"
#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of event fragments 
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/03/07 15:31:32 $
   */

  typedef ConcurrentQueue<I2OChain> FragmentQueue;
  typedef boost::shared_ptr<FragmentQueue> FragmentQueuePtr;
  
} // namespace stor

#endif // EventFilter_StorageManager_FragmentQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
