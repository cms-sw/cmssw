// $Id: DrainingQueues.cc,v 1.14 2011/11/08 10:48:40 mommsen Exp $
/// @file: DrainingQueues.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

DrainingQueues::DrainingQueues( my_context c ): my_base(c)
{
  safeEntryAction();
}

void DrainingQueues::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
}

DrainingQueues::~DrainingQueues()
{
  safeExitAction();
}

void DrainingQueues::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string DrainingQueues::do_stateName() const
{
  return std::string( "DrainingQueues" );
}

void DrainingQueues::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
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
    EventPtr_t stMachEvent( new QueuesEmpty() );
    sharedResources->commandQueue_->enqWait( stMachEvent );
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

  if ( ! sharedResources->streamQueue_->empty() ) return false;

  if ( sharedResources->diskWriterResources_->isBusy() ) return false;
  
  //if ( ! sharedResources->dqmEventQueue_->empty() ) return false;
  // Do not wait for dqmEventQueue to drain, just clear it
  sharedResources->dqmEventQueue_->clear();

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
      outermost_context().getFragmentStore()->getStaleEvent(staleEvent, boost::posix_time::seconds(0));
    if ( gotStaleEvent )
    {
      outermost_context().getSharedResources()->discardManager_->sendDiscardMessage(staleEvent);
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
