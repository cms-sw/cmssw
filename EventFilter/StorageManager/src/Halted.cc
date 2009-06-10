// $Id$

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include <iostream>

using namespace std;
using namespace stor;

Halted::Halted( my_context c ): my_base(c)
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  // Remi May 14, 2009: not clear why we originally introduced the _initialized
  // outermost_context().declareInitialized();
  outermost_context().setExternallyVisibleState( "Halted" );
  outermost_context().getNotifier()->reportNewState( "Halted" );
}

Halted::~Halted()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Halted::do_stateName() const
{
  return string( "Halted" );
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
