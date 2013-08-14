// $Id: StateMachineMonitorCollection.cc,v 1.9 2011/03/07 15:31:32 mommsen Exp $
/// @file: StateMachineMonitorCollection.cc

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StateMachineMonitorCollection.h"


namespace stor {
  
  StateMachineMonitorCollection::StateMachineMonitorCollection(const utils::Duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  externallyVisibleState_( "unknown" ),
  stateName_( "unknown" )
  {}
  

  void StateMachineMonitorCollection::updateHistory( const TransitionRecord& tr )
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    history_.push_back( tr );
  }
  
  
  void StateMachineMonitorCollection::getHistory( History& history ) const
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    history = history_;
  }
  
  
  void StateMachineMonitorCollection::dumpHistory( std::ostream& os ) const
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    
    os << "**** Begin transition history ****" << std::endl;
    
    for( History::const_iterator j = history_.begin();
         j != history_.end(); ++j )
    {
      os << "  " << *j << std::endl;
    }
    
    os << "**** End transition history ****" << std::endl;
    
  }
  
  
  void StateMachineMonitorCollection::setExternallyVisibleState( const std::string& n )
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    externallyVisibleState_ = n;
  }
  
  
  const std::string& StateMachineMonitorCollection::externallyVisibleState() const
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    return externallyVisibleState_;
  }
  
  
  void StateMachineMonitorCollection::setStatusMessage( const std::string& m )
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    if ( statusMessage_.empty() )
      statusMessage_ = m;
  }
  
  
  void StateMachineMonitorCollection::clearStatusMessage()
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    statusMessage_.clear();
  }
  
  
  bool StateMachineMonitorCollection::statusMessage( std::string& m ) const
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    m = statusMessage_;
    return ( ! statusMessage_.empty() );
  }
  
  
  std::string StateMachineMonitorCollection::innerStateName() const
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    TransitionRecord tr = history_.back();
    return tr.stateName();;
  }
  
  
  void StateMachineMonitorCollection::do_calculateStatistics()
  {
    // nothing to do
  }
  
  
  void StateMachineMonitorCollection::do_reset()
  {
    // we shall not reset the state name
    boost::mutex::scoped_lock sl( stateMutex_ );
    history_.clear();
  }
  
  
  void StateMachineMonitorCollection::do_appendInfoSpaceItems
  (
    InfoSpaceItems& infoSpaceItems
  )
  {
    infoSpaceItems.push_back(std::make_pair("stateName", &stateName_));
  }
  
  
  void StateMachineMonitorCollection::do_updateInfoSpaceItems()
  {
    boost::mutex::scoped_lock sl( stateMutex_ );
    
    stateName_ = static_cast<xdata::String>( externallyVisibleState_ );
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
