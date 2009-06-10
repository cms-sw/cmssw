/**
 * $Id$
 */

#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"


namespace stor
{

  void SharedResources::moveToFailedState()
  {
    event_ptr stMachEvent( new Fail() );
    // do we really want enq_wait here?
    // it could cause deadlock if the command queue is full...
    _commandQueue->enq_wait( stMachEvent );
  }  

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
