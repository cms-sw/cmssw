// $Id: RegistrationCollection.cc,v 1.4 2009/07/20 13:07:28 mommsen Exp $
/// @file: RegistrationCollection.cc

#include "EventFilter/StorageManager/interface/RegistrationCollection.h"

#include <boost/pointer_cast.hpp>

using namespace stor;

RegistrationCollection::RegistrationCollection()
{
  boost::mutex::scoped_lock sl( _lock );
  _nextConsumerID = ConsumerID(0);
  _registrationAllowed = false;
}

RegistrationCollection::~RegistrationCollection() {}

ConsumerID RegistrationCollection::getConsumerID()
{

  boost::mutex::scoped_lock sl( _lock );

  if( !_registrationAllowed )
    {
      return ConsumerID(0);
    }

  _nextConsumerID++;
  return _nextConsumerID;

}

bool
RegistrationCollection::addRegistrationInfo( ConsumerID cid, RegPtr ri )
{
  boost::mutex::scoped_lock sl( _lock );
  if( _registrationAllowed )
    {
      RegistrationMap::iterator pos = _consumers.lower_bound(cid);

      // 05-Oct-2009, KAB - added a test of whether cid appears before
      // pos->first in the map sort order.  Since lower_bound can return
      // a non-end iterator even when cid is not in the map, we need to
      // complete the test of whether cid is in the map.  (Another way to
      // look at this is we need to implement the full test described in
      // the efficientAddOrUpdates pattern suggested by Item 24 of
      // 'Effective STL' by Scott Meyers.)
      if ( pos != _consumers.end() && !(_consumers.key_comp()(cid, pos->first)) )
      {
        // The given ConsumerID already exists.
        return false;
      }

      _consumers.insert( pos, RegistrationMap::value_type(cid, ri) );
      return true;
    }
  else
    {
      return false;
    }
}

void RegistrationCollection::getEventConsumers( ConsumerRegistrations& crs )
{
  boost::mutex::scoped_lock sl( _lock );
  for( RegistrationMap::const_iterator it = _consumers.begin();
       it != _consumers.end(); ++it )
    {
      ConsRegPtr eventConsumer =
        boost::dynamic_pointer_cast<EventConsumerRegistrationInfo>( it->second );
      if ( eventConsumer )
        crs.push_back( eventConsumer );
    }
}

void RegistrationCollection::getDQMEventConsumers( DQMConsumerRegistrations& crs )
{
  boost::mutex::scoped_lock sl( _lock );
  for( RegistrationMap::const_iterator it = _consumers.begin();
       it != _consumers.end(); ++it )
    {
      DQMEventConsRegPtr dqmEventConsumer =
        boost::dynamic_pointer_cast<DQMEventConsumerRegistrationInfo>( it->second );
      if ( dqmEventConsumer )
        crs.push_back( dqmEventConsumer );
    }
}

void RegistrationCollection::enableConsumerRegistration()
{
  //boost::mutex::scoped_lock sl( _lock );
  _registrationAllowed = true;
}

void RegistrationCollection::disableConsumerRegistration()
{
  //boost::mutex::scoped_lock sl( _lock );
  _registrationAllowed = false;
}

void RegistrationCollection::clearRegistrations()
{
  boost::mutex::scoped_lock sl( _lock );
  _consumers.clear();
}

bool RegistrationCollection::registrationIsAllowed() const
{
  //boost::mutex::scoped_lock sl( _lock );
  return _registrationAllowed;
}

bool RegistrationCollection::isProxy( ConsumerID cid ) const
{

  boost::mutex::scoped_lock sl( _lock );

  RegistrationMap::const_iterator pos = _consumers.lower_bound(cid);

  // 05-Oct-2009, KAB - added a test of whether cid appears before
  // pos->first in the map sort order.  Since lower_bound can return
  // a non-end iterator even when cid is not in the map, we need to
  // complete the test of whether cid is in the map.  (Another way to
  // look at this is we need to implement the full test described in
  // the efficientAddOrUpdates pattern suggested by Item 24 of
  // 'Effective STL' by Scott Meyers.)
  if ( pos == _consumers.end() || (_consumers.key_comp()(cid, pos->first)) )
  {
    return false;
  }

  ConsRegPtr eventConsumer =
    boost::dynamic_pointer_cast<EventConsumerRegistrationInfo>( pos->second );
  if ( ! eventConsumer ) return false;

  return eventConsumer->isProxyServer();

}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
