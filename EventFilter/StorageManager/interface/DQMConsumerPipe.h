#ifndef STOR_DQMCONSUMER_PIPE_H
#define STOR_DQMCONSUMER_PIPE_H

/**
 * This class is used to manage the subscriptions, DQMevents, and
 * lost connections associated with an DQMevent consumer within the
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
 *  Initial Implementation based on Kurt's ConsumerPipe
 *  We can think about a common class later...
 *  $Id: DQMConsumerPipe.h,v 1.5 2007/11/09 23:08:34 badgett Exp $
 */

#include <string>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "EventFilter/StorageManager/interface/SMCurlInterface.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include "curl/curl.h"
#include <deque>

namespace stor
{
  class DQMConsumerPipe
  {
  public:
    DQMConsumerPipe(std::string name, std::string priority,
                    int activeTimeout, int idleTimeout,
                    std::string folderName, std::string hostName,
                    int queueSize);

    ~DQMConsumerPipe();

    uint32 getConsumerId() const;
    void initializeSelection();
    bool isIdle() const;
    bool isDisconnected() const;
    bool isReadyForEvent() const;
    bool wantsDQMEvent(DQMEventMsgView const& eventView) const;
    void putDQMEvent(boost::shared_ptr< std::vector<char> > bufPtr);
    boost::shared_ptr< std::vector<char> > getDQMEvent();
    void setPushMode(bool mode) { pushMode_ = mode; }
    void clearQueue();
    std::string getConsumerName() { return(consumerName_);}
    int getPushEventFailures() { return(pushEventFailures_);}
    int getEvents() { return(events_);}
    time_t getLastEventRequestTime() { return(lastEventRequestTime_);}
    std::string getHostName() { return(hostName_);}

  private:

    CURL* han_;
    struct curl_slist *headers_;
    // characteristics of the consumer
    uint32 consumerId_;
    std::string consumerName_;
    std::string consumerPriority_;
    std::string topFolderName_;
    std::string hostName_;
    int events_;

    // data members for tracking active and idle states
    int timeToIdleState_;          // seconds
    int timeToDisconnectedState_;  // seconds
    time_t lastEventRequestTime_;

    // track whether initialization has been completed
    bool initializationDone;

    // track if this consumer is a push-mode (SMProxyServer), name = URL
    bool pushMode_;
    bool pushEvent();
    unsigned int pushEventFailures_;

    // 28-Nov-2007, KAB: upgrade to a queue of events
    std::deque< boost::shared_ptr< std::vector<char> > > eventQueue_;
    unsigned int maxQueueSize_;

    // lock for controlling access to the event queue
    boost::mutex eventQueueLock_;

    // class data members used for creating unique consumer IDs
    static uint32 rootId_;
    static boost::mutex rootIdLock_;
  };
}

#endif
