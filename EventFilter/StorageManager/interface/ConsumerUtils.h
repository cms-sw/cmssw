// $Id: ConsumerUtils.h,v 1.5 2010/02/16 10:49:52 mommsen Exp $
/// @file: ConsumerUtils.h 

#ifndef StorageManager_ConsumerUtils_h
#define StorageManager_ConsumerUtils_h

#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
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
  class ConsumerID;
  class I2OChain;


  /**
     Handles consumer requests and responses

     $Author: mommsen $
     $Revision: 1.5 $
     $Date: 2010/02/16 10:49:52 $
  */

  class ConsumerUtils
  {
    
  public:

    void setSharedResources(SharedResourcesPtr sr)
    { _sharedResources = sr; };

    /**
      Process registration request from an event consumer
    */
    void processConsumerRegistrationRequest(xgi::Input*, xgi::Output*) const;

    /**
      Process header (init msg) request from an event consumer
    */
    void processConsumerHeaderRequest(xgi::Input*, xgi::Output*) const;

    /**
      Process event request from an event consumer
    */
    void processConsumerEventRequest(xgi::Input*, xgi::Output*) const;

    /**
      Process registration request from a DQM event (histogram) consumer
    */
    void processDQMConsumerRegistrationRequest(xgi::Input*, xgi::Output*) const;

    /**
      Process DQM event (histogram) request from a DQM event consumer
    */
    void processDQMConsumerEventRequest(xgi::Input*, xgi::Output*) const;


  private:

    /**
      Parse consumer registration request:
    */
    ConsRegPtr parseEventConsumerRegistration(xgi::Input*) const;

    /**
      Parse DQM consumer registration request:
    */
    DQMEventConsRegPtr parseDQMEventConsumerRegistration(xgi::Input*) const;

    /**
      Send ID to consumer:
    */
    void writeConsumerRegistration(xgi::Output*, const ConsumerID) const;

    /**
      Tell consumer we're not ready:
    */
    void writeNotReady(xgi::Output*) const;

    /**
      Send empty buffer to consumer:
    */
    void writeEmptyBuffer(xgi::Output*) const;

    /**
      Send a "done" message to consumer:
    */
    void writeDone(xgi::Output*) const;

    /**
      Send an error message to consumer:
    */
    void writeErrorString(xgi::Output*, const std::string) const;

    /**
      Write HTTP headers:
    */
    void writeHTTPHeaders(xgi::Output*) const;

    /**
      Extract consumer ID from header request:
    */
    ConsumerID getConsumerID(xgi::Input*) const;

    /**
      Send header to consumer:
    */
    void writeConsumerHeader(xgi::Output*, const InitMsgSharedPtr) const;

    /**
      Send event to consumer:
    */
    void writeConsumerEvent(xgi::Output*, const I2OChain&) const;

    /**
      Send DQM event to DQM consumer:
    */
    void writeDQMConsumerEvent(xgi::Output*, const DQMEventMsgView&) const;


    SharedResourcesPtr _sharedResources;

  };
}

#endif // StorageManager_ConsumerUtils_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
