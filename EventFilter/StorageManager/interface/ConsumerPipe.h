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
 * $Id: ConsumerPipe.h,v 1.13 2008/04/16 16:10:20 biery Exp $
 */

#include <string>
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/StorageManager/interface/SMCurlInterface.h"
#include "EventFilter/StorageManager/interface/ForeverCounter.h"
#include "EventFilter/StorageManager/interface/RollingIntervalCounter.h"
#include "EventFilter/StorageManager/interface/RollingSampleCounter.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include "curl/curl.h"
#include <deque>

namespace stor
{

  static const std::string PROXY_SERVER_NAME("SMProxyServer");

  class ConsumerPipe
  {
  public:
    enum STATS_TIME_FRAME { SHORT_TERM = 0, LONG_TERM = 1 };
    enum STATS_SAMPLE_TYPE { QUEUED_EVENTS = 10, SERVED_EVENTS = 11,
                             DESIRED_EVENTS = 12 };

    static const double MAX_ACCEPT_INTERVAL;

    ConsumerPipe(std::string name, std::string priority,
                 int activeTimeout, int idleTimeout,
                 Strings triggerSelection, double rateRequest,
                 std::string hltOutputSelection,
                 std::string hostName, int queueSize);

    ~ConsumerPipe();

    uint32 getConsumerId() const;
    void initializeSelection(Strings const& fullTriggerList, uint32 outputModuleId);
    bool isActive() const;
    bool isIdle() const;
    bool isDisconnected() const;
    bool isReadyForEvent(double currentTime = BaseCounter::getCurrentTime()) const;
    bool isProxyServer() const { return consumerIsProxyServer_; }
    bool hasRegistryWarning() const { return registryWarningWasReported_; }
    bool wantsEvent(EventMsgView const& eventView) const;
    void wasConsidered(double currentTime = BaseCounter::getCurrentTime());
    void putEvent(boost::shared_ptr< std::vector<char> > bufPtr);
    boost::shared_ptr< std::vector<char> > getEvent();
    void setPushMode(bool mode) { pushMode_ = mode; }
    void clearQueue();
    std::string getConsumerName() { return(consumerName_);}
    unsigned int getPushEventFailures() { return(pushEventFailures_);}
    unsigned int getEvents() { return(events_);}
    time_t getLastEventRequestTime() { return(lastEventRequestTime_);}
    std::string getHostName() { return(hostName_);}
    std::vector<std::string> getTriggerRequest() const;
    void setRegistryWarning(std::string const& message);
    void setRegistryWarning(std::vector<char> const& message);
    std::vector<char> getRegistryWarning() { return registryWarningMessage_; }
    void clearRegistryWarning() { registryWarningWasReported_ = false; }

    long long getEventCount(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                            STATS_SAMPLE_TYPE sampleType = QUEUED_EVENTS,
                            double currentTime = BaseCounter::getCurrentTime());
    double getEventRate(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                        STATS_SAMPLE_TYPE sampleType = QUEUED_EVENTS,
                        double currentTime = BaseCounter::getCurrentTime());
    double getDataRate(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                       STATS_SAMPLE_TYPE sampleType = QUEUED_EVENTS,
                       double currentTime = BaseCounter::getCurrentTime());
    double getDuration(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                       STATS_SAMPLE_TYPE sampleType = QUEUED_EVENTS,
                       double currentTime = BaseCounter::getCurrentTime());
    double getAverageQueueSize(STATS_TIME_FRAME timeFrame = SHORT_TERM,
                               STATS_SAMPLE_TYPE sampleType = QUEUED_EVENTS,
                               double currentTime = BaseCounter::getCurrentTime());
    Strings getTriggerSelection() const { return triggerSelection_; }
    double getRateRequest() const { return rateRequest_; }
    std::string getHLTOutputSelection() const { return hltOutputSelection_; }

  private:

    CURL* han_;
    struct curl_slist *headers_;
    // characteristics of the consumer
    uint32 consumerId_;
    std::string consumerName_;
    std::string consumerPriority_;
    int events_;
    Strings triggerSelection_;
    double rateRequest_;
    std::string hltOutputSelection_;
    uint32 hltOutputModuleId_;
    boost::shared_ptr<RollingSampleCounter> rateRequestCounter_;
    double minTimeBetweenEvents_;
    double lastConsideredEventTime_;
    std::string hostName_;
    bool consumerIsProxyServer_;

    // event selector that does the work of accepting/rejecting events
    boost::shared_ptr<edm::EventSelector> eventSelector_;

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

    // track whether a registry warning has been received
    bool registryWarningWasReported_;
    std::vector<char> registryWarningMessage_;

    // 28-Nov-2007, KAB: upgrade to a queue of events
    std::deque< boost::shared_ptr< std::vector<char> > > eventQueue_;
    unsigned int maxQueueSize_;

    // lock for controlling access to the event queue
    boost::mutex eventQueueLock_;

    // class data members used for creating unique consumer IDs
    static uint32 rootId_;
    static boost::mutex rootIdLock_;

    // statistics
    boost::shared_ptr<ForeverCounter> longTermDesiredCounter_;
    boost::shared_ptr<RollingIntervalCounter> shortTermDesiredCounter_;
    boost::shared_ptr<ForeverCounter> longTermQueuedCounter_;
    boost::shared_ptr<RollingIntervalCounter> shortTermQueuedCounter_;
    boost::shared_ptr<ForeverCounter> longTermServedCounter_;
    boost::shared_ptr<RollingIntervalCounter> shortTermServedCounter_;

    boost::shared_ptr<ForeverCounter> ltQueueSizeWhenDesiredCounter_;
    boost::shared_ptr<RollingIntervalCounter> stQueueSizeWhenDesiredCounter_;
    boost::shared_ptr<ForeverCounter> ltQueueSizeWhenQueuedCounter_;
    boost::shared_ptr<RollingIntervalCounter> stQueueSizeWhenQueuedCounter_;
  };
}

#endif
