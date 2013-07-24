// $Id: StateMachine.cc,v 1.7 2011/04/18 15:18:57 mommsen Exp $
/// @file: StateMachine.cc

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/Notifier.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <typeinfo>
#include <fstream>


namespace stor {
  
  StateMachine::StateMachine
  ( 
    EventDistributor* ed,
    FragmentStore* fs,
    Notifier* n,
    SharedResourcesPtr sr
  ):
  eventDistributor_(ed),
  fragmentStore_(fs),
  notifier_(n),
  sharedResources_(sr)
  {}
  
  Operations const&
  StateMachine::getCurrentState() const
  {
    return state_cast<Operations const&>();
  }
  
  std::string StateMachine::getCurrentStateName() const
  {
    return getCurrentState().stateName();
  }
  
  void StateMachine::updateHistory( const TransitionRecord& tr )
  {
    sharedResources_->statisticsReporter_->
      getStateMachineMonitorCollection().updateHistory(tr);
    // std::ostringstream msg;
    // msg << tr;
    // LOG4CPLUS_WARN(sharedResources_->statisticsReporter_->alarmHandler()->getLogger(), msg.str());
  }
  
  void StateMachine::unconsumed_event( bsc::event_base const &event )
  {
    
    std::cerr << "The " << 
      //event.dynamic_type()
      typeid(event).name()
      << " event is not supported from the "
      << getCurrentStateName() << " state!" << std::endl;
    
    // Tell run control not to wait:
    notifier_->reportNewState( "Unchanged" );
  }
  
  void StateMachine::setExternallyVisibleState( const std::string& s )
  {
    sharedResources_->statisticsReporter_->
      getStateMachineMonitorCollection().setExternallyVisibleState( s );
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
