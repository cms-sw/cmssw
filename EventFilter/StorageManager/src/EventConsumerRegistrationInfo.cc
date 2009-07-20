// $Id: EventConsumerRegistrationInfo.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $
/// @file: EventConsumerRegistrationInfo.cc

#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include <algorithm>
#include <iterator>
#include <ostream>

using namespace std;

namespace stor
{

  EventConsumerRegistrationInfo::EventConsumerRegistrationInfo
  ( const unsigned int& maxConnectRetries,
    const unsigned int& connectRetryInterval, // seconds
    const string& consumerName,
    const FilterList& selEvents,
    const string& selHLTOut,
    const size_t& queueSize,
    const enquing_policy::PolicyTag& queuePolicy,
    const utils::duration_t& secondsToStale ) :
  _common( consumerName, queueSize, queuePolicy, secondsToStale ),
  _maxConnectRetries( maxConnectRetries ),
  _connectRetryInterval( connectRetryInterval ),
  _selEvents( selEvents ),
  _selHLTOut( selHLTOut ),
  _stale( false )
  {
    if( consumerName == "SMProxyServer" ||
        ( consumerName.find( "urn" ) != std::string::npos &&
          consumerName.find( "xdaq" ) != std::string::npos &&
          consumerName.find( "pushEventData" ) != std::string::npos ) )
      {
        _isProxy = true;
      }
    else
      {
        _isProxy = false;
      }
  }

  EventConsumerRegistrationInfo::~EventConsumerRegistrationInfo()
  { }

  void 
  EventConsumerRegistrationInfo::do_registerMe(EventDistributor* evtDist)
  {
    evtDist->registerEventConsumer(this);
  }

  QueueID
  EventConsumerRegistrationInfo::do_queueId() const
  {
    return _common._queueId;
  }

  void
  EventConsumerRegistrationInfo::do_setQueueID(QueueID const& id)
  {
    _common._queueId = id;
  }

  string
  EventConsumerRegistrationInfo::do_consumerName() const
  {
    return _common._consumerName;
  }

  ConsumerID
  EventConsumerRegistrationInfo::do_consumerId() const
  {
    return _common._consumerId;
  }

  void
  EventConsumerRegistrationInfo::do_setConsumerID(ConsumerID const& id)
  {
    _common._consumerId = id;
  }

  size_t
  EventConsumerRegistrationInfo::do_queueSize() const
  {
    return _common._queueSize;
  }

  enquing_policy::PolicyTag
  EventConsumerRegistrationInfo::do_queuePolicy() const
  {
    return _common._queuePolicy;
  }

  utils::duration_t
  EventConsumerRegistrationInfo::do_secondsToStale() const
  {
    return _common._secondsToStale;
  }


  ostream& 
  EventConsumerRegistrationInfo::write(ostream& os) const
  {
    os << "EventConsumerRegistrationInfo:"
       << _common
       << "\n Maximum number of connection attempts: "
       << _maxConnectRetries
       << "\n Connection retry interval, seconds: "
       << _connectRetryInterval
       << "\n HLT output: " << _selHLTOut
       << "\n Event filters:\n";

    copy(_selEvents.begin(), 
         _selEvents.end(),
         ostream_iterator<FilterList::value_type>(os, "\n"));
    
    //     for( unsigned int i = 0; i < _selEvents.size(); ++i )
    //       {
    //         os << '\n' << "  " << _selEvents[i];
    //       }

    return os;
  }

}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
