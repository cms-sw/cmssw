// $Id: Running.cc,v 1.11 2011/11/08 10:48:41 mommsen Exp $
/// @file: Running.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

Running::Running( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Running::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // Configure event distributor
  EventDistributor* ed = outermost_context().getEventDistributor();
  EvtStrConfigListPtr evtCfgList = sharedResources->configuration_->
    getCurrentEventStreamConfig();
  ErrStrConfigListPtr errCfgList = sharedResources->configuration_->
    getCurrentErrorStreamConfig();
  ed->registerEventStreams(evtCfgList);
  ed->registerErrorStreams(errCfgList);

  // Clear old consumer registrations:
  sharedResources->registrationCollection_->clearRegistrations();
  ed->clearConsumers();
  sharedResources->eventQueueCollection_->removeQueues();
  sharedResources->dqmEventQueueCollection_->removeQueues();

  // Enable consumer registration:
  sharedResources->registrationCollection_->enableConsumerRegistration();
}

Running::~Running()
{
  safeExitAction();
}

void Running::do_exitActionWork()
{

  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // Disable consumer registration:
  sharedResources->registrationCollection_->disableConsumerRegistration();

  // Clear consumer queues
  sharedResources->eventQueueCollection_->clearQueues();
  sharedResources->dqmEventQueueCollection_->clearQueues();

  // Clear the queues
  sharedResources->fragmentQueue_->clear();
  sharedResources->streamQueue_->clear();
  sharedResources->dqmEventQueue_->clear();
  sharedResources->registrationQueue_->clear();

  // Clear any fragments left in the fragment store
  outermost_context().getFragmentStore()->clear();

}

string Running::do_stateName() const
{
  return std::string( "Running" );
}

void Running::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
}

void Running::logStopDoneRequest( const StopDone& request )
{
  outermost_context().unconsumed_event( request );
}

void Running::logHaltDoneRequest( const HaltDone& request )
{
  outermost_context().unconsumed_event( request );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
