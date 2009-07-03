// $Id: Enabled.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include <iostream>

using namespace std;
using namespace stor;

Enabled::Enabled( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Enabled::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // reset all statistics (needs to be done first)
  sharedResources->_statisticsReporter->reset();

  // clear the INIT message collection at begin run
  sharedResources->_initMsgCollection->clear();

  // update the run-based configuration parameters
  sharedResources->_configuration->updateRunParams();
}

Enabled::~Enabled()
{
  safeExitAction( outermost_context().getNotifier() );
}

void Enabled::do_exitActionWork()
{

  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );

  // clear the stream selections in the event distributor
  outermost_context().getEventDistributor()->clearStreams();

}

string Enabled::do_stateName() const
{
  return string( "Enabled" );
}

void Enabled::logReconfigureRequest( const Reconfigure& request )
{
  outermost_context().unconsumed_event( request );
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
