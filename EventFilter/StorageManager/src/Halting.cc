// $Id: Halting.cc,v 1.2 2009/06/10 08:15:27 dshpakov Exp $

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"


#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

Halting::Halting( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Halting::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // request that the streams that are currently configured in the disk
  // writer be destroyed (this has the side effect of closing files)
  sharedResources->_diskWriterResources->requestStreamDestruction();

  // request that the DQM event store is cleared
  // if FinishingDQM has succeeded, the store is already empty
  sharedResources->_dqmEventProcessorResources->requestStoreDestruction();
}

Halting::~Halting()
{
  safeExitAction( outermost_context().getNotifier() );
}

void Halting::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Halting::do_stateName() const
{
  return string( "Halting" );
}

void Halting::logStopDoneRequest( const StopDone& request )
{
  outermost_context().unconsumed_event( request );
}

void
Halting::do_noFragmentToProcess() const
{
  if ( destructionIsDone() )
  {
    SharedResourcesPtr sharedResources =
      outermost_context().getSharedResources();
    event_ptr stMachEvent( new HaltDone() );
    sharedResources->_commandQueue->enq_wait( stMachEvent );
  }
}

bool
Halting::destructionIsDone() const
{
  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // check if the requests are still being processed
  if ( sharedResources->_diskWriterResources->streamChangeOngoing() ) return false;

  if ( sharedResources->_dqmEventProcessorResources->requestsOngoing() ) return false;

  return true;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
