// $Id: RegistrationCollection.cc,v 1.11 2011/03/07 15:31:32 mommsen Exp $
/// @file: RegistrationCollection.cc

#include "EventFilter/StorageManager/interface/RegistrationCollection.h"

#include <boost/pointer_cast.hpp>
#include <algorithm>

using namespace stor;

RegistrationCollection::RegistrationCollection()
{
  boost::mutex::scoped_lock sl( lock_ );
  nextConsumerId_ = ConsumerID(0);
  registrationAllowed_ = false;
}

RegistrationCollection::~RegistrationCollection() {}

ConsumerID RegistrationCollection::getConsumerId()
{
  boost::mutex::scoped_lock sl( lock_ );
  
  if( !registrationAllowed_ )
  {
    return ConsumerID(0);
  }
  
  return ++nextConsumerId_;
}

bool
RegistrationCollection::addRegistrationInfo( const RegPtr ri )
{
  boost::mutex::scoped_lock sl( lock_ );
  if( registrationAllowed_ )
  {
    ConsumerID cid = ri->consumerId();
    RegistrationMap::iterator pos = consumers_.lower_bound(cid);
    
    if ( pos != consumers_.end() && !(consumers_.key_comp()(cid, pos->first)) )
    {
      // The given ConsumerID already exists.
      return false;
    }
    
    consumers_.insert( pos, RegistrationMap::value_type(cid, ri) );
    return true;
  }
  else
  {
    return false;
  }
}


RegPtr RegistrationCollection::getRegistrationInfo( const ConsumerID cid ) const
{
  boost::mutex::scoped_lock sl( lock_ );
  RegPtr regInfo;
  RegistrationMap::const_iterator pos = consumers_.find(cid);
  if ( pos != consumers_.end() )
  {
    pos->second->consumerContact();
    regInfo = pos->second;
  }
  return regInfo;
}


void RegistrationCollection::getEventConsumers( ConsumerRegistrations& crs ) const
{
  boost::mutex::scoped_lock sl( lock_ );
  for( RegistrationMap::const_iterator it = consumers_.begin();
       it != consumers_.end(); ++it )
    {
      EventConsRegPtr eventConsumer =
        boost::dynamic_pointer_cast<EventConsumerRegistrationInfo>( it->second );
      if ( eventConsumer )
        crs.push_back( eventConsumer );
    }
  // sort the event consumers to have identical consumers sharing a queue
  // next to each others.
  utils::ptrComp<EventConsumerRegistrationInfo> comp;
  std::sort(crs.begin(), crs.end(), comp);
}

void RegistrationCollection::getDQMEventConsumers( DQMConsumerRegistrations& crs ) const
{
  boost::mutex::scoped_lock sl( lock_ );
  for( RegistrationMap::const_iterator it = consumers_.begin();
       it != consumers_.end(); ++it )
    {
      DQMEventConsRegPtr dqmEventConsumer =
        boost::dynamic_pointer_cast<DQMEventConsumerRegistrationInfo>( it->second );
      if ( dqmEventConsumer )
        crs.push_back( dqmEventConsumer );
    }
}

void RegistrationCollection::enableConsumerRegistration()
{
  //boost::mutex::scoped_lock sl( lock_ );
  registrationAllowed_ = true;
}

void RegistrationCollection::disableConsumerRegistration()
{
  //boost::mutex::scoped_lock sl( lock_ );
  registrationAllowed_ = false;
}

void RegistrationCollection::clearRegistrations()
{
  boost::mutex::scoped_lock sl( lock_ );
  consumers_.clear();
}

bool RegistrationCollection::registrationIsAllowed( const ConsumerID cid ) const
{
  boost::mutex::scoped_lock sl( lock_ );

  RegistrationMap::const_iterator pos = consumers_.find(cid);
  if ( pos == consumers_.end() ) return false;
  pos->second->consumerContact();

  return registrationAllowed_;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
