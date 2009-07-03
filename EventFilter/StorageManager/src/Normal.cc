// $Id: Normal.cc,v 1.2 2009/06/10 08:15:27 dshpakov Exp $

#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include <iostream>

using namespace std;
using namespace stor;

void Normal::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
}

Normal::Normal( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Normal::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

Normal::~Normal()
{
  safeExitAction( outermost_context().getNotifier() );
}

string Normal::do_stateName() const
{
  return string( "Normal" );
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
