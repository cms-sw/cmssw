// $Id: StateMachineMonitorCollection.cc,v 1.4 2009/07/10 09:19:17 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StateMachineMonitorCollection.h"

using namespace stor;

StateMachineMonitorCollection::StateMachineMonitorCollection() :
MonitorCollection(),
_externallyVisibleState( "unknown" ),
_statusMessageAvailable( false ),
_statusMessage( "" ),
_stateName( "unknown" )
{}


void StateMachineMonitorCollection::updateHistory( const TransitionRecord& tr )
{
  boost::mutex::scoped_lock sl( _stateMutex );
  _history.push_back( tr );
}


void StateMachineMonitorCollection::getHistory( History& history ) const
{
  boost::mutex::scoped_lock sl( _stateMutex );
  history = _history;
}


void StateMachineMonitorCollection::dumpHistory( std::ostream& os ) const
{
  boost::mutex::scoped_lock sl( _stateMutex );

  os << "**** Begin transition history ****" << std::endl;

  for( History::const_iterator j = _history.begin();
       j != _history.end(); ++j )
    {
      os << "  " << *j << std::endl;
    }

  os << "**** End transition history ****" << std::endl;

}


void StateMachineMonitorCollection::setExternallyVisibleState( const std::string& n )
{
  boost::mutex::scoped_lock sl( _stateMutex );
  _externallyVisibleState = n;
}


const std::string& StateMachineMonitorCollection::externallyVisibleState() const
{
  boost::mutex::scoped_lock sl( _stateMutex );
  return _externallyVisibleState;
}



void StateMachineMonitorCollection::setStatusMessage( const std::string& m )
{
  boost::mutex::scoped_lock sl( _stateMutex );
  _statusMessageAvailable = true;
  _statusMessage = m;
}


void StateMachineMonitorCollection::clearStatusMessage()
{
  boost::mutex::scoped_lock sl( _stateMutex );
  _statusMessageAvailable = false;
  _statusMessage = "";
}


bool StateMachineMonitorCollection::statusMessage( std::string& m ) const
{
  boost::mutex::scoped_lock sl( _stateMutex );
  m = _statusMessage;
  return _statusMessageAvailable;
}


const std::string& StateMachineMonitorCollection::innerStateName() const
{
  boost::mutex::scoped_lock sl( _stateMutex );
  TransitionRecord tr = _history.back();
  return tr.stateName();;
}


void StateMachineMonitorCollection::do_calculateStatistics()
{
  // nothing to do
}


void StateMachineMonitorCollection::do_reset()
{
  // we shall not reset the state name
  boost::mutex::scoped_lock sl( _stateMutex );
  _history.clear();
}


void StateMachineMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("stateName", &_stateName));
}


void StateMachineMonitorCollection::do_updateInfoSpaceItems()
{
  boost::mutex::scoped_lock sl( _stateMutex );

  _stateName = static_cast<xdata::String>( _externallyVisibleState );
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
