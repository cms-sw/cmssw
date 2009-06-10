// $Id$

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

void stor::sm_debug( const std::string& file_name_suffix, const std::string& message )
{
  const std::string fname = std::string( "/tmp/storage_manager_debug_" ) + file_name_suffix;
  std::ofstream f( fname.c_str() );
  if( f.is_open() )
    {
      f << message << std::endl;
      f.close();
    }
}


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
// Remi May 14, 2009: not clear why we originally introduced the _initialized
// _initialized( false )
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
// Remi May 14, 2009: not clear why we originally introduced the _initialized
//   if( _initialized )
//     {
//       if( _sharedResources->_statisticsReporter.get() != 0 )
//         {
          _sharedResources->_statisticsReporter->
            getStateMachineMonitorCollection().setExternallyVisibleState( s );
//         }
//     }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
