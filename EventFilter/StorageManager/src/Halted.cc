// $Id: Halted.cc,v 1.4 2009/07/03 14:13:04 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include <iostream>

#include "xcept/tools.h"

using namespace std;
using namespace stor;

void Halted::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  outermost_context().setExternallyVisibleState( "Halted" );
  outermost_context().getNotifier()->reportNewState( "Halted" );
}

Halted::Halted( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Halted::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

Halted::~Halted()
{
  safeExitAction( outermost_context().getNotifier() );
}

string Halted::do_stateName() const
{
  return string( "Halted" );
}

void Halted::do_moveToFailedState() const
{
  outermost_context().getSharedResources()->moveToFailedState();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
