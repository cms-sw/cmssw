// $Id: Halting.cc,v 1.11 2011/11/08 10:48:41 mommsen Exp $
/// @file: Halting.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"


#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

Halting::Halting( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Halting::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // request that the streams that are currently configured in the disk
  // writer be destroyed (this has the side effect of closing files)
  sharedResources->diskWriterResources_->requestStreamDestruction();

  // request that the DQM event store is cleared
  // if FinishingDQM has succeeded, the store is already empty
  sharedResources->dqmEventProcessorResources_->requestStoreDestruction();
}

Halting::~Halting()
{
  safeExitAction();
}

void Halting::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Halting::do_stateName() const
{
  return std::string( "Halting" );
}

void Halting::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
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
    EventPtr_t stMachEvent( new HaltDone() );
    sharedResources->commandQueue_->enqWait( stMachEvent );
  }
}

bool
Halting::destructionIsDone() const
{
  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // check if the requests are still being processed
  if ( sharedResources->diskWriterResources_->streamChangeOngoing() ) return false;

  if ( sharedResources->dqmEventProcessorResources_->requestsOngoing() ) return false;

  return true;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
