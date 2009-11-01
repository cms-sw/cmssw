// $Id: Failed.cc,v 1.10.4.1 2009/09/25 09:57:48 mommsen Exp $
/// @file: Failed.cc

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>

using namespace std;
using namespace stor;

void Failed::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  outermost_context().setExternallyVisibleState( "Failed" );
  outermost_context().getNotifier()->reportNewState( "Failed" );

  // request that the streams that are currently configured in the disk
  // writer be destroyed (this has the side effect of closing files).
  // This should have been done by the Halting/Stopping entry actions,
  // but if we Fail, we need to do it here. No harm if we do it twice.
  outermost_context().getSharedResources()->
    _diskWriterResources->requestStreamDestruction();
}

Failed::Failed( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Failed::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

Failed::~Failed()
{
  safeExitAction();
}

string Failed::do_stateName() const
{
  return string( "Failed" );
}

void Failed::do_moveToFailedState( xcept::Exception& exception ) const
{
  // nothing can be done here
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
