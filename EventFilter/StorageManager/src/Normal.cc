// $Id$

#include "EventFilter/StorageManager/interface/StateMachine.h"

#include <iostream>

using namespace std;
using namespace stor;

Normal::Normal( my_context c ): my_base(c)
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );
}

Normal::~Normal()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
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
