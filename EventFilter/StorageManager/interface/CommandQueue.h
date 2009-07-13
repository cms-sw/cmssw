#ifndef EventFilter_StorageManager_CommandQueue_h
#define EventFilter_StorageManager_CommandQueue_h

#include "boost/statechart/event_base.hpp"
#include "boost/shared_ptr.hpp"
#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"

/**
   Concurrent queue holding state machine events

   $ Author: $
   $ Revision: $
   $ Date: $
*/

namespace stor
{
  typedef boost::shared_ptr<boost::statechart::event_base> event_ptr;
  typedef ConcurrentQueue<event_ptr> CommandQueue;
}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
