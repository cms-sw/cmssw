// $Id: DQMEventQueue.h,v 1.5 2011/03/07 15:31:31 mommsen Exp $
/// @file: DQMEventQueue.h 

#ifndef EventFilter_StorageManager_DQMEventQueue_h
#define EventFilter_StorageManager_DQMEventQueue_h

#include "boost/shared_ptr.hpp"
#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
   * Queue holding I2OChains of complete DQM events (histograms)
   * waiting to be processed by the DQMEventProcessor
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2011/03/07 15:31:31 $
   */

  typedef ConcurrentQueue< I2OChain, KeepNewest<I2OChain> > DQMEventQueue;  
  typedef boost::shared_ptr<DQMEventQueue> DQMEventQueuePtr;

} // namespace stor

#endif // EventFilter_StorageManager_DQMEventQueue_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
