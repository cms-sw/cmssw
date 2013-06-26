// $Id: CommandQueue.h,v 1.6 2011/03/07 15:31:31 mommsen Exp $
/// @file: CommandQueue.h 

#ifndef EventFilter_StorageManager_CommandQueue_h
#define EventFilter_StorageManager_CommandQueue_h

#include "boost/statechart/event_base.hpp"
#include "boost/shared_ptr.hpp"
#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"

namespace stor
{

  /**
     Concurrent queue holding state machine events

     $Author: mommsen $
     $Revision: 1.6 $
     $Date: 2011/03/07 15:31:31 $
  */
  typedef boost::shared_ptr<boost::statechart::event_base> EventPtr_t;
  typedef ConcurrentQueue<EventPtr_t> CommandQueue;
  typedef boost::shared_ptr<CommandQueue> CommandQueuePtr;

} // namespace stor

#endif // EventFilter_StorageManager_CommandQueue_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
