// $Id: Enabled.cc,v 1.12 2011/11/08 10:48:40 mommsen Exp $
/// @file: Enabled.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>

using namespace std;
using namespace stor;

Enabled::Enabled( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Enabled::do_entryActionWork()
{

  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // reset all statistics (needs to be done first)
  sharedResources->statisticsReporter_->reset();

  // clear the INIT message collection at begin run
  sharedResources->initMsgCollection_->clear();

  // update the run-based configuration parameters
  sharedResources->configuration_->updateRunParams();
}

Enabled::~Enabled()
{
  safeExitAction();
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
  return std::string( "Enabled" );
}

void Enabled::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
}

void Enabled::logHaltRequest( const Halt& request )
{
  outermost_context().unconsumed_event( request );
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
