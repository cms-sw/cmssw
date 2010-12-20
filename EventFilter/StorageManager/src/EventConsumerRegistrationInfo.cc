// $Id: EventConsumerRegistrationInfo.cc,v 1.13 2010/12/17 18:21:05 mommsen Exp $
/// @file: EventConsumerRegistrationInfo.cc

#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include <algorithm>
#include <iterator>
#include <ostream>


namespace stor
{

  EventConsumerRegistrationInfo::EventConsumerRegistrationInfo
  ( const std::string& consumerName,
    const std::string& triggerSelection,
    const Strings& eventSelection,
    const std::string& outputModuleLabel,
    const unsigned int& prescale,
    const bool& uniqueEvents,
    const int& queueSize,
    const enquing_policy::PolicyTag& queuePolicy,
    const utils::duration_t& secondsToStale,
    const std::string& remoteHost ) :
    _common( consumerName, queueSize, queuePolicy, secondsToStale ),
    _triggerSelection( triggerSelection ),
    _eventSelection( eventSelection ),
    _outputModuleLabel( outputModuleLabel ),
    _prescale( prescale ),
    _uniqueEvents( uniqueEvents ),
    _stale( false ),
    _remoteHost( remoteHost )
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
  EventConsumerRegistrationInfo::do_setQueueId(QueueID const& id)
  {
    _common._queueId = id;
  }

  std::string
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
  EventConsumerRegistrationInfo::do_setConsumerId(ConsumerID const& id)
  {
    _common._consumerId = id;
  }

  int
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

  bool
  EventConsumerRegistrationInfo::operator<(const EventConsumerRegistrationInfo& other) const
  {
    if ( _outputModuleLabel != other.outputModuleLabel() )
      return ( _outputModuleLabel < other.outputModuleLabel() );
    if ( _triggerSelection != other.triggerSelection() )
      return ( _triggerSelection < other.triggerSelection() );
    if ( _eventSelection != other.eventSelection() )
      return ( _eventSelection < other.eventSelection() );
    if ( _prescale != other.prescale() )
      return ( _prescale < other.prescale() );
    if ( _uniqueEvents != other.uniqueEvents() )
      return ( _uniqueEvents < other.uniqueEvents() );
    if ( _common._queueSize != other.queueSize() )
      return ( _common._queueSize < other.queueSize() );
    if ( _common._queuePolicy != other.queuePolicy() )
      return ( _common._queuePolicy < other.queuePolicy() );
    return ( _common._secondsToStale < other.secondsToStale() );
  }

  bool
  EventConsumerRegistrationInfo::operator==(const EventConsumerRegistrationInfo& other) const
  {
    return (
      _outputModuleLabel == other.outputModuleLabel() &&
      _triggerSelection == other.triggerSelection() &&
      _eventSelection == other.eventSelection() &&
      _prescale == other.prescale() &&
      _uniqueEvents == other.uniqueEvents() &&
      _common._queueSize == other.queueSize() &&
      _common._queuePolicy == other.queuePolicy() &&
      _common._secondsToStale == other.secondsToStale()
    );
  }

  bool
  EventConsumerRegistrationInfo::operator!=(const EventConsumerRegistrationInfo& other) const
  {
    return ! ( *this == other );
  }

  std::ostream& 
  EventConsumerRegistrationInfo::write(std::ostream& os) const
  {
    os << "EventConsumerRegistrationInfo:"
       << _common
       << "\n HLT output: " << _outputModuleLabel
       << "\n Event filters:\n";
    /*
    if (_triggerSelection.size()) {
      os << std::endl << _triggerSelection;
    }
    else 
    */
    std::copy(_eventSelection.begin(), 
              _eventSelection.end(),
              std::ostream_iterator<Strings::value_type>(os, "\n"));
    
    //     for( unsigned int i = 0; i < _eventSelection.size(); ++i )
    //       {
    //         os << '\n' << "  " << _eventSelection[i];
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
