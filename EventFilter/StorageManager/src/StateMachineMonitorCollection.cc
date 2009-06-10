// $Id$

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/StateMachineMonitorCollection.h"

using namespace stor;

StateMachineMonitorCollection::StateMachineMonitorCollection(xdaq::Application *app) :
MonitorCollection(app),
_logger(app->getApplicationLogger()),
_externallyVisibleState( "unknown" ),
_stateName( "unknown" )
{
  _infoSpaceItems.push_back(std::make_pair("stateName", &_stateName));

  putItemsIntoInfoSpace();
}

void StateMachineMonitorCollection::updateHistory( const TransitionRecord& tr )
{
  std::ostringstream msg;
  if ( tr.isEntry() )
    msg << "Entered state ";
  else
    msg << "Exited state ";
  msg << tr.stateName();
  //  LOG4CPLUS_INFO(_logger, msg.str());

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



void StateMachineMonitorCollection::do_calculateStatistics()
{
  // nothing to do
}


void StateMachineMonitorCollection::do_updateInfoSpace()
{
  boost::mutex::scoped_lock sl( _stateMutex );

  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();

    _stateName = static_cast<xdata::String>( _externallyVisibleState );

    _infoSpace->unlock();
  }
  catch(std::exception &e)
  {
    _infoSpace->unlock();
 
    errorMsg += ": ";
    errorMsg += e.what();
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }
  catch (...)
  {
    _infoSpace->unlock();
 
    errorMsg += " : unknown exception";
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }

  try
  {
    // The fireItemGroupChanged locks the infospace
    _infoSpace->fireItemGroupChanged(_infoSpaceItemNames, this);
  }
  catch (xdata::exception::Exception &e)
  {
    XCEPT_RETHROW(stor::exception::Monitoring, errorMsg, e);
  }
}


void StateMachineMonitorCollection::do_reset()
{
  // we shall not reset the state name
  boost::mutex::scoped_lock sl( _stateMutex );
  _history.clear();
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


const std::string& StateMachineMonitorCollection::innerStateName() const
{
  boost::mutex::scoped_lock sl( _stateMutex );
  TransitionRecord tr = _history.back();
  return tr.stateName();;
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
