// $Id: DQMEventConsumerRegistrationInfo.cc,v 1.2 2009/06/10 08:15:25 dshpakov Exp $
/// @file: DQMEventConsumerRegistrationInfo.cc

#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"

using namespace std;

namespace stor
{

  DQMEventConsumerRegistrationInfo::DQMEventConsumerRegistrationInfo
  ( const std::string& consumerName,
    const string& topLevelFolderName,
    const size_t& queueSize,
    const enquing_policy::PolicyTag& queuePolicy,
    const utils::duration_t& secondsToStale ) :
  _common( consumerName, queueSize, queuePolicy, secondsToStale),
  _topLevelFolderName( topLevelFolderName ),
  _stale( false )
  {
    if( consumerName == "SMProxyServer" ||
        ( consumerName.find( "urn" ) != std::string::npos &&
          consumerName.find( "xdaq" ) != std::string::npos &&
          consumerName.find( "pushDQMEventData" ) != std::string::npos ) )
      {
        _isProxy = true;
      }
    else
      {
        _isProxy = false;
      }
  }

  DQMEventConsumerRegistrationInfo::~DQMEventConsumerRegistrationInfo() 
  { }

  void 
  DQMEventConsumerRegistrationInfo::do_registerMe(EventDistributor* evtDist)
  {
    evtDist->registerDQMEventConsumer(this);
  }

  QueueID
  DQMEventConsumerRegistrationInfo::do_queueId() const
  {
    return _common._queueId;
  }

  void
  DQMEventConsumerRegistrationInfo::do_setQueueID(QueueID const& id)
  {
    _common._queueId = id;
  }

  string
  DQMEventConsumerRegistrationInfo::do_consumerName() const
  {
    return _common._consumerName;
  }

  ConsumerID
  DQMEventConsumerRegistrationInfo::do_consumerId() const
  {
    return _common._consumerId;
  }

  void
  DQMEventConsumerRegistrationInfo::do_setConsumerID(ConsumerID const& id)
  {
    _common._consumerId = id;
  }

  size_t
  DQMEventConsumerRegistrationInfo::do_queueSize() const
  {
    return _common._queueSize;
  }

  enquing_policy::PolicyTag
  DQMEventConsumerRegistrationInfo::do_queuePolicy() const
  {
    return _common._queuePolicy;
  }

  utils::duration_t
  DQMEventConsumerRegistrationInfo::do_secondsToStale() const
  {
    return _common._secondsToStale;
  }

  ostream&
  DQMEventConsumerRegistrationInfo::write(ostream& os) const
  {
    os << "DQMEventConsumerRegistrationInfo:"
       << _common
       << "\n Top folder name: " << _topLevelFolderName;
    return os;
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
