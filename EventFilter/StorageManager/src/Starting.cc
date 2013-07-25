// $Id: Starting.cc,v 1.12 2011/11/08 10:48:41 mommsen Exp $
/// @file: Starting.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

Starting::Starting( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Starting::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // Request DiskWriter to configure streams
  EvtStrConfigListPtr evtCfgList = sharedResources->configuration_->
    getCurrentEventStreamConfig();
  ErrStrConfigListPtr errCfgList = sharedResources->configuration_->
    getCurrentErrorStreamConfig();

  WorkerThreadParams workerParams =
    sharedResources->configuration_->getWorkerThreadParams();
  sharedResources->diskWriterResources_->
    requestStreamConfiguration(evtCfgList, errCfgList,
      sharedResources->configuration_->getDiskWritingParams(),
      sharedResources->configuration_->getRunNumber(),
      workerParams.DWdeqWaitTime_);

  // Request configuration of DQMEventProcessor
  sharedResources->dqmEventProcessorResources_->
    requestConfiguration(
      sharedResources->configuration_->getDQMProcessingParams(),
      workerParams.DQMEPdeqWaitTime_);
}

Starting::~Starting()
{
  safeExitAction();
}

void Starting::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Starting::do_stateName() const
{
  return std::string( "Starting" );
}

void Starting::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
}

void Starting::logStopDoneRequest( const StopDone& request )
{
  outermost_context().unconsumed_event( request );
}

void Starting::logHaltDoneRequest( const HaltDone& request )
{
  outermost_context().unconsumed_event( request );
}

void
Starting::do_noFragmentToProcess() const
{
  if ( workerThreadsConfigured() )
  {
    SharedResourcesPtr sharedResources =
      outermost_context().getSharedResources();
    EventPtr_t stMachEvent( new StartRun() );
    sharedResources->commandQueue_->enqWait( stMachEvent );
  }
}

bool
Starting::workerThreadsConfigured() const
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
