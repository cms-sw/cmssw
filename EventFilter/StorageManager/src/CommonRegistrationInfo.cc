// $Id: CommonRegistrationInfo.cc,v 1.4 2010/02/16 10:49:36 mommsen Exp $
/// @file: CommonRegistrationInfo.cc

#include "EventFilter/StorageManager/interface/CommonRegistrationInfo.h"

using std::string;

namespace stor
{
  CommonRegistrationInfo::CommonRegistrationInfo
  (
    const std::string& consumerName,
    const int& queueSize,
    const enquing_policy::PolicyTag& queuePolicy,
    const utils::duration_t& secondsToStale
  ) :
  _consumerName(consumerName),
  _queueSize(queueSize),
  _queuePolicy(queuePolicy),
  _secondsToStale(secondsToStale),
  _consumerId(0)
  { }

  std::ostream& operator<< (std::ostream& os,
                            CommonRegistrationInfo const& ri)
  {
    os << "\n Consumer name: " << ri._consumerName
       << "\n Consumer id: " << ri._consumerId
       << "\n Queue id: " << ri._queueId
       << "\n Maximum size of queue: " << ri._queueSize
       << "\n Policy used if queue is full: " << ri._queuePolicy
       << "\n Time until queue becomes stale (seconds): " << ri._secondsToStale.total_seconds();
    return os;
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
