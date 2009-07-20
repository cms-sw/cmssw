// $Id: StateMachine.cc,v 1.3 2009/07/03 19:30:55 mommsen Exp $
/// @file: StateMachine.cc

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <typeinfo>
#include <fstream>

using namespace stor;
using namespace std;


StateMachine::StateMachine
( 
  EventDistributor* ed,
  FragmentStore* fs,
  Notifier* n,
  SharedResourcesPtr sr
):
_eventDistributor(ed),
_fragmentStore(fs),
_notifier(n),
_sharedResources(sr)
{
}

Operations const&
StateMachine::getCurrentState() const
{
  return state_cast<Operations const&>();
}

string StateMachine::getCurrentStateName() const
{
  return getCurrentState().stateName();
}

void StateMachine::updateHistory( const TransitionRecord& tr )
{
  _sharedResources->_statisticsReporter->
    getStateMachineMonitorCollection().updateHistory(tr);
}

void StateMachine::unconsumed_event( bsc::event_base const &event )
{

  std::cerr << "The " << 
    //event.dynamic_type()
    typeid(event).name()
    << " event is not supported from the "
    << getCurrentStateName() << " state!" << std::endl;

  // Tell run control not to wait:
  _notifier->reportNewState( "Unchanged" );

}

void StateMachine::setExternallyVisibleState( const std::string& s )
{
  _sharedResources->_statisticsReporter->
    getStateMachineMonitorCollection().setExternallyVisibleState( s );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
