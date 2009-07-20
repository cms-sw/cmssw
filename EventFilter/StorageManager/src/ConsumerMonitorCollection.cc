// $Id: ConsumerMonitorCollection.cc,v 1.3 2009/07/09 15:34:28 mommsen Exp $
/// @file: ConsumerMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"

using namespace stor;


ConsumerMonitorCollection::ConsumerMonitorCollection():
  MonitorCollection()
{}


void ConsumerMonitorCollection::addQueuedEventSample( QueueID qid,
						      unsigned int data_size )
{
  boost::mutex::scoped_lock l( _mutex );
  if( _qmap.find( qid ) != _qmap.end() )
    {
      _qmap[ qid ]->addSample( data_size );
    }
  else
    {
      _qmap[ qid ] = boost::shared_ptr<MonitoredQuantity>( new MonitoredQuantity() );
      _qmap[ qid ]->addSample( data_size );
    }
}


void ConsumerMonitorCollection::addServedEventSample( QueueID qid,
						      unsigned int data_size )
{
  boost::mutex::scoped_lock l( _mutex );
  if( _smap.find( qid ) != _smap.end() )
    {
      _smap[ qid ]->addSample( data_size );
    }
  else
    {
      _smap[ qid ] = boost::shared_ptr<MonitoredQuantity>( new MonitoredQuantity() );
      _smap[ qid ]->addSample( data_size );
    }
}


bool ConsumerMonitorCollection::getQueued( QueueID qid,
					   MonitoredQuantity::Stats& result )
{
  boost::mutex::scoped_lock l( _mutex );
  if( _qmap.find( qid ) == _qmap.end() ) return false;
  _qmap[ qid ]->getStats( result );
  return true;
}


bool ConsumerMonitorCollection::getServed( QueueID qid,
					   MonitoredQuantity::Stats& result )
{
  boost::mutex::scoped_lock l( _mutex );
  if( _smap.find( qid ) == _smap.end() ) return false;
  _smap[ qid ]->getStats( result );
  return true;
}


void ConsumerMonitorCollection::resetCounters()
{
  boost::mutex::scoped_lock l( _mutex );
  for( ConsStatMap::iterator i = _qmap.begin(); i != _qmap.end(); ++i )
    i->second->reset();
  for( ConsStatMap::iterator i = _smap.begin(); i != _smap.end(); ++i )
    i->second->reset();
}


void ConsumerMonitorCollection::do_calculateStatistics()
{
  boost::mutex::scoped_lock l( _mutex );
  for( ConsStatMap::iterator i = _qmap.begin(); i != _qmap.end(); ++i )
    i->second->calculateStatistics();
  for( ConsStatMap::iterator i = _smap.begin(); i != _smap.end(); ++i )
    i->second->calculateStatistics();
}


void ConsumerMonitorCollection::do_reset()
{
  boost::mutex::scoped_lock l( _mutex );
  _qmap.clear();
  _smap.clear();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
