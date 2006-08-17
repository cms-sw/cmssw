#ifndef STOR_EVENT_SERVER_H
#define STOR_EVENT_SERVER_H

/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include <string>
#include <vector>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"

namespace stor
{
  class EventServer
  {
  public:
    EventServer(double maximumRate, int vipConsumerQueueSize,
                double consumerIdleTime, double consumerDisconnectTime);
    ~EventServer();

    void addConsumer(boost::shared_ptr<ConsumerPipe> consumer);
    bool wantsEvent(const EventMsgView &eventView);
    //void putEvent(const EventMsgView &eventView);
    // processEvent??
    //boost::shared_ptr<EventMsgView> getEvent(int consumerId);

  private:
    double maximumTotalRate_;  // per second
    int vipQueueSize_;
    double consumerIdleTime_;  // seconds
    double consumerDisconnectTime_;  // seconds

    std::vector< boost::shared_ptr<ConsumerPipe> > consumerList;
    //std::vector<boost::shared_ptr<ConsumerPipe>> vipConsumerList;
  };
}

#endif
