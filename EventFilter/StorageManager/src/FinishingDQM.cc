// $Id: FinishingDQM.cc,v 1.9 2009/09/29 07:57:56 mommsen Exp $
/// @file: FinishingDQM.cc

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace stor;

FinishingDQM::FinishingDQM( my_context c ): my_base(c)
{
  safeEntryAction();
}

void FinishingDQM::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // request end-of-run processing in DQMEventProcessor
  sharedResources->_dqmEventProcessorResources->requestEndOfRun();

}

FinishingDQM::~FinishingDQM()
{
  safeExitAction();
}

void FinishingDQM::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string FinishingDQM::do_stateName() const
{
  return std::string( "FinishingDQM" );
}

void FinishingDQM::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->moveToFailedState( exception );
}

void
FinishingDQM::do_noFragmentToProcess() const
{
  if ( endOfRunProcessingIsDone() )
  {
    SharedResourcesPtr sharedResources =
      outermost_context().getSharedResources();
    event_ptr stMachEvent( new EndRun() );
    sharedResources->_commandQueue->enq_wait( stMachEvent );
  }
}

bool
FinishingDQM::endOfRunProcessingIsDone() const
{
  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  if ( sharedResources->_dqmEventProcessorResources->requestsOngoing() ) return false; 

  return true;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
