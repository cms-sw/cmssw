// $Id: Normal.cc,v 1.10 2011/11/08 10:48:41 mommsen Exp $
/// @file: Normal.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

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
  safeEntryAction();
}

void Normal::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

Normal::~Normal()
{
  safeExitAction();
}

string Normal::do_stateName() const
{
  return std::string( "Normal" );
}

void Normal::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
