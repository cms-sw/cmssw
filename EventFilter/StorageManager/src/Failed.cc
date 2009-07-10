// $Id: Failed.cc,v 1.8 2009/07/03 19:31:19 mommsen Exp $

#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include <iostream>

using namespace std;
using namespace stor;

void Failed::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
  outermost_context().setExternallyVisibleState( "Failed" );
  outermost_context().getNotifier()->reportNewState( "Failed" );
}

Failed::Failed( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
}

void Failed::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

Failed::~Failed()
{
  safeExitAction( outermost_context().getNotifier() );
}

string Failed::do_stateName() const
{
  return string( "Failed" );
}

void Failed::do_moveToFailedState( const std::string& reason ) const
{
  // nothing can be done here
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
