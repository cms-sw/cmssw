// $Id: DrainingQueues.cc,v 1.4 2009/07/03 19:31:19 mommsen Exp $

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

DrainingQueues::DrainingQueues( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void DrainingQueues::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
}

DrainingQueues::~DrainingQueues()
{
  safeExitAction( outermost_context().getNotifier() );
}

void DrainingQueues::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string DrainingQueues::do_stateName() const
{
  return string( "DrainingQueues" );
}

void DrainingQueues::do_moveToFailedState() const
{
  outermost_context().getSharedResources()->moveToFailedState();
}

void DrainingQueues::logEndRunRequest( const EndRun& request )
{
  outermost_context().unconsumed_event( request );
}

void
DrainingQueues::do_noFragmentToProcess() const
{
  if ( allQueuesAndWorkersAreEmpty() )
  {
    SharedResourcesPtr sharedResources =
      outermost_context().getSharedResources();
    event_ptr stMachEvent( new QueuesEmpty() );
    sharedResources->_commandQueue->enq_wait( stMachEvent );
  }
}

bool
DrainingQueues::allQueuesAndWorkersAreEmpty() const
{
  SharedResourcesPtr sharedResources = 
    outermost_context().getSharedResources();

  // the order is important here - upstream entities first,
  // followed by more downstream entities

  EventDistributor *ed = outermost_context().getEventDistributor();
  if ( ed->full() ) return false;

  processStaleFragments();
  FragmentStore *fs = outermost_context().getFragmentStore();
  if ( ! fs->empty() ) return false;

  if ( ! sharedResources->_streamQueue->empty() ) return false;

  if ( sharedResources->_diskWriterResources->isBusy() ) return false;
  
  if ( ! sharedResources->_dqmEventQueue->empty() ) return false;

  return true;
}

void
DrainingQueues::processStaleFragments() const
{
  I2OChain staleEvent;
  bool gotStaleEvent = true;  
  int loopCounter = 0;

  EventDistributor *ed = outermost_context().getEventDistributor();

  while ( gotStaleEvent && !ed->full() && loopCounter++ < 10 )
  {
    gotStaleEvent = 
      outermost_context().getFragmentStore()->getStaleEvent(staleEvent, 0);
    if ( gotStaleEvent )
    {
      outermost_context().getSharedResources()->_discardManager->sendDiscardMessage(staleEvent);
      ed->addEventToRelevantQueues(staleEvent);
    }
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
