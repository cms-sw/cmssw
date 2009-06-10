// $Id$

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include <iostream>

using namespace std;
using namespace stor;

Failed::Failed( my_context c ): my_base(c)
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  outermost_context().setExternallyVisibleState( "Failed" );
  outermost_context().getNotifier()->reportNewState( "Failed" );
}

Failed::~Failed()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Failed::do_stateName() const
{
  return string( "Failed" );
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
