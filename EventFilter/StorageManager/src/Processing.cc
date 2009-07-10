// $Id: Processing.cc,v 1.8 2009/07/03 19:31:19 mommsen Exp $

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include <iostream>
#include <sstream>

#include "xcept/tools.h"

using namespace std;
using namespace stor;

Processing::Processing( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Processing::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  outermost_context().setExternallyVisibleState( "Enabled" );
  outermost_context().getNotifier()->reportNewState( "Enabled" );
}

Processing::~Processing()
{
  safeExitAction( outermost_context().getNotifier() );
}

void Processing::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Processing::do_stateName() const
{
  return string( "Processing" );
}

void Processing::do_moveToFailedState() const
{
  outermost_context().getSharedResources()->moveToFailedState();
}

void Processing::logEndRunRequest( const EndRun& request )
{
  outermost_context().unconsumed_event( request );
}

void
Processing::do_processI2OFragment( I2OChain& frag ) const
{
  static unsigned int noFragmentCount = 0;

  bool completed = outermost_context().getFragmentStore()->addFragment(frag);
  if ( completed )
  {
    outermost_context().getSharedResources()->_discardManager->sendDiscardMessage(frag);

    try
    {
      uint32 runNumber = outermost_context().getSharedResources()->_configuration->getRunNumber();
      frag.assertRunNumber(runNumber);
    }
    catch(stor::exception::RunNumberMismatch &e)
    {
      outermost_context().getEventDistributor()->addEventToRelevantQueues(frag);
      XCEPT_RETHROW(stor::exception::RunNumberMismatch, e.message(), e);
    }
    outermost_context().getEventDistributor()->addEventToRelevantQueues(frag);
  }
  else
  {
    // Only do the check every 100th fragment
    // TODO: shall we make this number configurable?
    ++noFragmentCount;
    if ( noFragmentCount >= 100 )
    {
      noFragmentCount = 0;
      this->noFragmentToProcess();
    }
  }
}

void
Processing::do_noFragmentToProcess() const
{
  I2OChain staleEvent;

  WorkerThreadParams workerParams =
    outermost_context().getSharedResources()->_configuration->getWorkerThreadParams();
  bool gotStaleEvent = 
    outermost_context().getFragmentStore()->
    getStaleEvent(staleEvent, workerParams._staleFragmentTimeOut);

  if ( gotStaleEvent )
  {
    outermost_context().getSharedResources()->_discardManager->sendDiscardMessage(staleEvent);
    outermost_context().getEventDistributor()->addEventToRelevantQueues(staleEvent);
  }
  outermost_context().getEventDistributor()->checkForStaleConsumers();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
