// $Id: ConsumerUtils.h,v 1.10 2011/08/22 14:20:16 mommsen Exp $
/// @file: ConsumerUtils.h 

#ifndef EventFilter_StorageManager_ConsumerUtils_h
#define EventFilter_StorageManager_ConsumerUtils_h

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <boost/shared_ptr.hpp>

namespace xgi
{
  class Input;
  class Output;
}

namespace smproxy
{
  class EventMsg;
}

namespace stor
{
  class ConsumerID;


  /**
     Handles consumer requests and responses

     $Author: mommsen $
     $Revision: 1.10 $
     $Date: 2011/08/22 14:20:16 $
  */

  template<typename Configuration_t, typename EventQueueCollection_t>
  class ConsumerUtils
  {
    
  public:

    ConsumerUtils
    (
      boost::shared_ptr<Configuration_t>,
      RegistrationCollectionPtr,
      RegistrationQueuePtr,
      InitMsgCollectionPtr,
      boost::shared_ptr<EventQueueCollection_t>,
      DQMEventQueueCollectionPtr,
      AlarmHandlerPtr
    );

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
      Create the event consumer registration info for this request
    */
    EventConsRegPtr createEventConsumerRegistrationInfo(xgi::Input*, xgi::Output*) const;

    /**
      Parse consumer registration request
    */
    EventConsRegPtr parseEventConsumerRegistration(xgi::Input*) const;

    /**
      Create an event consumer queue. Returns true on success.
    */
    bool createEventConsumerQueue(EventConsRegPtr) const;

    /**
      Create the DQM event consumer registration info for this request
    */
    DQMEventConsRegPtr createDQMEventConsumerRegistrationInfo(xgi::Input*, xgi::Output*) const;

    /**
      Parse DQM consumer registration request
    */
    DQMEventConsRegPtr parseDQMEventConsumerRegistration(xgi::Input*) const;

    /**
      Create a DQM event consumer queue. Returns true on success.
    */
    bool createDQMEventConsumerQueue(DQMEventConsRegPtr) const;

    /**
      Add registration information to RegistrationCollection and
      to registration queue. Return true on success.
    */
    bool addRegistrationInfo(const RegPtr) const;

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
    ConsumerID getConsumerId(xgi::Input*) const;

    /**
      Send header to consumer:
    */
    void writeConsumerHeader(xgi::Output*, const InitMsgSharedPtr) const;

    /**
      Send event to consumer:
    */
    void writeConsumerEvent(xgi::Output*, const stor::I2OChain&) const;
    void writeConsumerEvent(xgi::Output*, const smproxy::EventMsg&) const;

    /**
      Send DQM event to DQM consumer:
    */
    void writeDQMConsumerEvent(xgi::Output*, const DQMEventMsgView&) const;


    boost::shared_ptr<Configuration_t> configuration_;
    RegistrationCollectionPtr registrationCollection_;
    RegistrationQueuePtr registrationQueue_;
    InitMsgCollectionPtr initMsgCollection_;
    boost::shared_ptr<EventQueueCollection_t> eventQueueCollection_;
    DQMEventQueueCollectionPtr dqmEventQueueCollection_;
    AlarmHandlerPtr alarmHandler_;
  };

} // namespace stor

#endif // EventFilter_StorageManager_ConsumerUtils_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
