// $Id: ConsumerUtils.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: ConsumerUtils.h 

#ifndef StorageManager_ConsumerUtils_h
#define StorageManager_ConsumerUtils_h

#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <boost/shared_ptr.hpp>

namespace xgi
{
  class Input;
  class Output;
}

namespace stor
{

  /**
     Free helper functions for handling consumer header and event
     requests and responses

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */

  /**
     Parse consumer registration request:
  */
  ConsRegPtr parseEventConsumerRegistration( xgi::Input* in,
                                             size_t queueSize,
                                             enquing_policy::PolicyTag queuePolicy,
                                             utils::duration_t secondsToStale );

  /**
     Parse DQM consumer registration request:
  */
  DQMEventConsRegPtr parseDQMEventConsumerRegistration( xgi::Input* in,
                                                   size_t queueSize,
                                                   enquing_policy::PolicyTag queuePolicy,
                                                   utils::duration_t secondsToStale );

  /**
     Send ID to consumer:
  */
  void writeConsumerRegistration( xgi::Output*, ConsumerID );

  /**
     Tell consumer we're not ready:
  */
  void writeNotReady( xgi::Output* );

  /**
     Send empty buffer to consumer:
  */
  void writeEmptyBuffer( xgi::Output* );

  /**
     Send a "done" message to consumer:
  */
  void writeDone( xgi::Output* );

  /**
     Send an error message to consumer:
  */
  void writeErrorString( xgi::Output*, std::string );

  /**
     Write HTTP headers:
  */
  void writeHTTPHeaders( xgi::Output* );

  /**
     Extract consumer ID from header request:
  */
  ConsumerID getConsumerID( xgi::Input* );

  /**
     Send header to consumer:
  */
  void writeConsumerHeader( xgi::Output*, InitMsgSharedPtr );

  /**
     Send event to consumer:
  */
  void writeConsumerEvent( xgi::Output*, const I2OChain& );

  /**
     Send DQM event to DQM consumer:
  */
  void writeDQMConsumerEvent( xgi::Output*, const DQMEventMsgView& );

}

#endif // StorageManager_ConsumerUtils_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
