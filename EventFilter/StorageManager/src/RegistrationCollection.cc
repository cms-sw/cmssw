// $Id: RegistrationCollection.cc,v 1.9 2010/04/16 14:40:14 mommsen Exp $
/// @file: RegistrationCollection.cc

#include "EventFilter/StorageManager/interface/RegistrationCollection.h"

#include <boost/pointer_cast.hpp>

using namespace stor;

RegistrationCollection::RegistrationCollection()
{
  boost::mutex::scoped_lock sl( _lock );
  _nextConsumerId = ConsumerID(0);
  _registrationAllowed = false;
}

RegistrationCollection::~RegistrationCollection() {}

ConsumerID RegistrationCollection::getConsumerId()
{
  boost::mutex::scoped_lock sl( _lock );
  
  if( !_registrationAllowed )
  {
    return ConsumerID(0);
  }
  
  return ++_nextConsumerId;
}

bool
RegistrationCollection::addRegistrationInfo( const RegPtr ri )
{
  boost::mutex::scoped_lock sl( _lock );
  if( _registrationAllowed )
  {
    ConsumerID cid = ri->consumerId();
    RegistrationMap::iterator pos = _consumers.lower_bound(cid);
    
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


RegPtr RegistrationCollection::getRegistrationInfo( const ConsumerID cid ) const
{
  boost::mutex::scoped_lock sl( _lock );
  RegPtr regInfo;
  RegistrationMap::const_iterator pos = _consumers.find(cid);
  if ( pos != _consumers.end() )
  {
    regInfo = pos->second;
  }
  return regInfo;
}


void RegistrationCollection::getEventConsumers( ConsumerRegistrations& crs ) const
{
  boost::mutex::scoped_lock sl( _lock );
  for( RegistrationMap::const_iterator it = _consumers.begin();
       it != _consumers.end(); ++it )
    {
      EventConsRegPtr eventConsumer =
        boost::dynamic_pointer_cast<EventConsumerRegistrationInfo>( it->second );
      if ( eventConsumer )
        crs.push_back( eventConsumer );
    }
}

void RegistrationCollection::getDQMEventConsumers( DQMConsumerRegistrations& crs ) const
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


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
