#ifndef STOR_CONSUMER_PIPE_H
#define STOR_CONSUMER_PIPE_H

/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include <string>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

namespace stor
{
  class ConsumerPipe
  {
  public:
    ConsumerPipe(std::string name, std::string priority);
    ~ConsumerPipe();

    uint32 getConsumerId();
    //bool isReadyForEvent();
    //bool wantsEvent(const EventMsgView &eventView);
    //void putEvent(EventMsgView &eventView);
    //boost::shared_ptr<EventMsgView> getEvent();

  private:
    uint32 consumerId_;
    std::string consumerName_;
    std::string consumerPriority_;
    //long lastEventRequestTime_;

    // class data members used for creating unique consumer IDs
    static uint32 rootId_;
    static boost::mutex rootIdLock_;
  };
}

#endif
