#ifndef STOR_EVENT_SERVER_H
#define STOR_EVENT_SERVER_H

/**
 * This class manages the distribution of events to consumers from within
 * the storage manager.
 *
 * Two ways of throttling events are supported:
 * specifying a maximimum allowed rate of accepted events and specifying
 * a fixed prescale.  If the fixed prescale value is greater than zero,
 * it takes precendence.  That is, the maximum rate is ignored if the
 * prescale is in effect.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 * $Id: EventServer.h,v 1.6 2007/11/09 23:08:34 badgett Exp $
 */

#include <sys/time.h>
#include <string>
#include <vector>
#include <map>
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
    EventServer(double maximumRate);
    ~EventServer();

    void addConsumer(boost::shared_ptr<ConsumerPipe> consumer);
    std::map< uint32, boost::shared_ptr<ConsumerPipe> > getConsumerTable();
    boost::shared_ptr<ConsumerPipe> getConsumer(uint32 consumerId);

    void processEvent(const EventMsgView &eventView);
    boost::shared_ptr< std::vector<char> > getEvent(uint32 consumerId);
    void clearQueue();

    void setStreamSelectionTable(std::map<std::string, Strings> const& selTable);
    std::map<std::string, Strings> getStreamSelectionTable()
    {
      return streamSelectionTable_;
    }
    int getSelectionTableStringSize()
    {
      return selTableStringSize_;
    }
    Strings updateTriggerSelectionForStreams(Strings const& selectionList);

  private:
    // data members for handling a maximum rate of accepted events
    static const double MAX_ACCEPT_INTERVAL;
    double minTimeBetweenEvents_;  // seconds
    struct timeval lastAcceptedEventTime_;

    // data members for deciding when to check for disconnected consumers
    int disconnectedConsumerTestCounter_;

    // consumer lists
    std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable_;
    //std::vector<boost::shared_ptr<ConsumerPipe>> vipConsumerList;

    std::map<std::string, Strings> streamSelectionTable_;
    int selTableStringSize_;
  };
}

#endif
