#ifndef STOR_CONSUMER_PIPE_H
#define STOR_CONSUMER_PIPE_H

/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * In this initial version, a consumer is treated as being in one of
 * three states: active, idle, or disconnected. Operationally, these states
 * are meant to indicate the following:
 * - the consumer is alive and requesting events (active)
 * - the consumer has not requested an event in some time, but it is still
 *    connected to the storage manager (idle)
 * - the consumer is no longer connected (disconnected)
 * Since we don't have an actual connection to each consumer, we use
 * various timeouts to decide when the idle and disconnected states
 * should be reported. The logic used is the following:
 * - if ((now - lastEventRequestTime_) > activeTimeout), then
 *   isIdle() will return true. Otherwise, it will return false.
 * - if ((now - lastEventRequestTime_) > (idleTimeout + activeTimeout)), then
 *   isDisconnected() will return true. Otherwise, false.  (In this case,
 *   isIdle() will return false since the consumer has moved from the idle
 *   to the disconnected state.)
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
    ConsumerPipe(std::string name, std::string priority,
                 int activeTimeout, int idleTimeout);
    ~ConsumerPipe();

    uint32 getConsumerId() const;
    bool isIdle() const;
    bool isDisconnected() const;
    bool isReadyForEvent() const;
    bool wantsEvent(const EventMsgView &eventView) const;
    void putEvent(boost::shared_ptr< vector<char> > bufPtr);
    boost::shared_ptr< vector<char> > getEvent();

  private:
    // characteristics of the consumer
    uint32 consumerId_;
    std::string consumerName_;
    std::string consumerPriority_;

    // data members for tracking active and idle states
    int timeToIdleState_;          // seconds
    int timeToDisconnectedState_;  // seconds
    time_t lastEventRequestTime_;

    // for consumers with normal priority, we keep only the most recent event
    boost::shared_ptr< vector<char> > latestEvent_;

    // lock for controlling access to the most recent event
    boost::mutex latestEventLock_;

    // class data members used for creating unique consumer IDs
    static uint32 rootId_;
    static boost::mutex rootIdLock_;
  };
}

#endif
