// $Id: StateMachine.h,v 1.7 2009/07/14 10:34:44 dshpakov Exp $
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

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */
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
