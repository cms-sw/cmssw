/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 * $Id: ConsumerPipe.cc,v 1.19 2008/03/03 20:09:37 biery Exp $
 */

#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// keep this for debugging
//#include "IOPool/Streamer/interface/DumpTools.h"

using namespace std;
using namespace stor;
using namespace edm;

/**
 * Initialize the static value for the root consumer id.
 */
uint32 ConsumerPipe::rootId_ = 1;

/**
 * Initialize the static lock used to control access to the root ID.
 */
boost::mutex ConsumerPipe::rootIdLock_;

/**
 * Initialize the maximum accept interval.
 */
const double ConsumerPipe::MAX_ACCEPT_INTERVAL = 86400.0;  // seconds in 1 day

/**
 * ConsumerPipe constructor.
 */
ConsumerPipe::ConsumerPipe(std::string name, std::string priority,
                           int activeTimeout, int idleTimeout,
                           Strings triggerSelection, double rateRequest,
                           std::string hostName, int queueSize):
  han_(curl_easy_init()),
  headers_(),
  consumerName_(name),consumerPriority_(priority),
  events_(0),
  triggerSelection_(triggerSelection),
  rateRequest_(rateRequest),
  hostName_(hostName),
  pushEventFailures_(0),
  maxQueueSize_(queueSize)
{
  // initialize the time values we use for defining "states"
  timeToIdleState_ = activeTimeout;
  timeToDisconnectedState_ = activeTimeout + idleTimeout;
  lastEventRequestTime_ = time(NULL);
  initializationDone = false;
  pushMode_ = false;
  if(consumerPriority_.compare("PushMode") == 0) pushMode_ = true;
  registryWarningWasReported_ = false;

  // determine if we're connected to a proxy server
  consumerIsProxyServer_ = false;
  //if (consumerName_ == PROXY_SERVER_NAME)
  if (consumerName_.find("urn") != std::string::npos &&
      consumerName_.find("xdaq") != std::string::npos &&
      consumerName_.find("pushEventData") != std::string::npos)
  {
    consumerIsProxyServer_ = true;
  }

  // assign the consumer ID
  boost::mutex::scoped_lock scopedLockForRootId(rootIdLock_);
  consumerId_ = rootId_;
  rootId_++;

  if(han_==0)
  {
    edm::LogError("ConsumerPipe") << "Could not create curl handle";
    //std::cout << "Could not create curl handle" << std::endl;
    // throw exception here when we can make the SM go to a fail state from
    // another thread
  } else {
    headers_ = curl_slist_append(headers_, "Content-Type: application/octet-stream");
    headers_ = curl_slist_append(headers_, "Content-Transfer-Encoding: binary");
    // Avoid the Expect: 100 continue automatic header that gives a 2 sec delay
    // for pthttp but we don't need the Expect: 100 continue anyway
    headers_ = curl_slist_append(headers_, "Expect:");
    setopt(han_, CURLOPT_HTTPHEADER, headers_);
    setopt(han_, CURLOPT_URL, consumerName_.c_str());
    setopt(han_, CURLOPT_WRITEFUNCTION, func);
    // debug options
    //setopt(han_,CURLOPT_VERBOSE, 1);
    //setopt(han_,CURLOPT_TCP_NODELAY, 1);
  }

  // determine the amount of time that we need to wait between accepted
  // events.  The request rate specified to this constructor is
  // converted to an interval that is used internally, and the interval
  // is required to be somewhat reasonable.
  if (rateRequest_ < (1.0 / MAX_ACCEPT_INTERVAL))
  {
    minTimeBetweenEvents_ = MAX_ACCEPT_INTERVAL;
  }
  else
  {
    minTimeBetweenEvents_ = 1.0 / rateRequest_;  // seconds
  }
  lastConsideredEventTime_ = BaseCounter::getCurrentTime();
  rateRequestCounter_.reset(new RollingSampleCounter(50,1,60,RollingSampleCounter::INCLUDE_SAMPLES_IMMEDIATELY));

  // initialize the counters that we use for statistics
  longTermDesiredCounter_.reset(new ForeverCounter());
  shortTermDesiredCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermQueuedCounter_.reset(new ForeverCounter());
  shortTermQueuedCounter_.reset(new RollingIntervalCounter(180,5,20));
  longTermServedCounter_.reset(new ForeverCounter());
  shortTermServedCounter_.reset(new RollingIntervalCounter(180,5,20));

  ltQueueSizeWhenDesiredCounter_.reset(new ForeverCounter());
  stQueueSizeWhenDesiredCounter_.reset(new RollingIntervalCounter(180,5,20));
  ltQueueSizeWhenQueuedCounter_.reset(new ForeverCounter());
  stQueueSizeWhenQueuedCounter_.reset(new RollingIntervalCounter(180,5,20));
}

/**
 * ConsumerPipe destructor.
 */
ConsumerPipe::~ConsumerPipe()
{
  FDEBUG(5) << "Executing destructor for consumer pipe with ID = " <<
    consumerId_ << std::endl;
  curl_slist_free_all(headers_);
  curl_easy_cleanup(han_);
}

/**
 * Returns the consumer ID associated with this pipe.
 */
uint32 ConsumerPipe::getConsumerId() const
{
  return consumerId_;
}

/**
 * Initializes the event selection for this consumer based on the
 * specified full list of triggers and the request ParameterSet
 * that was specified in the constructor.
 */
void ConsumerPipe::initializeSelection(Strings const& fullTriggerList)
{
  FDEBUG(5) << "Initializing consumer pipe, ID = " <<
    consumerId_ << std::endl;

  // create our event selector
  eventSelector_.reset(new EventSelector(triggerSelection_,
					 fullTriggerList));
  // indicate that initialization is complete
  initializationDone = true;

}

/**
 * Tests if the consumer is active.  The active state indicates that
 * the consumer is connected and is actively requesting events.
 */
bool ConsumerPipe::isActive() const
{
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff < timeToIdleState_);
}

/**
 * Tests if the consumer is idle (as opposed to active).  The idle
 * state indicates that the consumer is still connected, but it hasn't
 * requested an event in some time.
 */
bool ConsumerPipe::isIdle() const
{
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff >= timeToIdleState_ &&
          timeDiff <  timeToDisconnectedState_);
}

/**
 * Tests if the consumer has disconnected.
 */
bool ConsumerPipe::isDisconnected() const
{
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff >= timeToDisconnectedState_);
}

/**
 * Tests if the consumer is ready for an event.  This method is often
 * used in conjunction with the wantsEvent() method.  In those cases,
 * the wantsEvent() method should be called first to get
 * "DESIRED" event statistics that are not biased by the consumer
 * rate request.
 */
bool ConsumerPipe::isReadyForEvent(double currentTime) const
{
  // we're not ready if we haven't yet been initialized
  // or are no longer active
  if (! initializationDone) {return false;}
  if (! this->isActive()) {return false;}

  // check if enough time has elapsed since the last event was considered.
  // 16-Apr-2008, KAB:  The simple method here would be to always use
  // "currentTime - lastTime" is greater-than-or-equal-to the minimum time
  // between events.  However, that doesn't provide enough rate, in my
  // opinion.  For example, let's say that the rate of triggers for a
  // particular consumer is 1.0 Hz, and the consumer rate request is 10 Hz.
  // The simple method will undershoot the 1.0 Hz because occasionally
  // an event is just under the 1.0 sec cutoff.  Using a longer time frame
  // (with a RollingSampleCounter or something else) allows us to provide
  // a more accurate requested rate.
  if (! rateRequestCounter_->hasValidResult()) {
    return ((currentTime - lastConsideredEventTime_) >= minTimeBetweenEvents_);
  }
  else {
    return (rateRequestCounter_->getSampleRate(currentTime) < rateRequest_);
  }
}

/**
 * Tests if the consumer wants the specified event.  This method is often
 * used in conjuntion with the isReadyForEvent() method.  In those cases,
 * this method should be called first to generate "DESIRED" event
 * statistics that are not biased by the consumer rate request.
 */
bool ConsumerPipe::wantsEvent(EventMsgView const& eventView) const
{
  // we're not interested in events if we haven't yet been initialized
  // or are no longer active
  if (! initializationDone) {return false;}
  if (! this->isActive()) {return false;}

  // get trigger bits for this event and check using eventSelector_
  std::vector<unsigned char> hlt_out;
  hlt_out.resize(1 + (eventView.hltCount()-1)/4);
  eventView.hltTriggerBits(&hlt_out[0]);
  int num_paths = eventView.hltCount();
  bool rc = (eventSelector_->wantAll() || eventSelector_->acceptEvent(&hlt_out[0], num_paths));

  // if we want this event, add it to our statistics for "desired"
  // or "acceptable" events.
  if (rc) {
    double now = BaseCounter::getCurrentTime();
    double sizeInMB = static_cast<double>(eventView.size()) / 1048576.0;
    ltQueueSizeWhenDesiredCounter_->addSample(eventQueue_.size());
    stQueueSizeWhenDesiredCounter_->addSample(eventQueue_.size(), now);
    longTermDesiredCounter_->addSample(sizeInMB);
    shortTermDesiredCounter_->addSample(sizeInMB, now);
  }
  return rc;
}

/**
 * Tells the ConsumerPipe that an event was considered or accepted or queued.
 *
 * This method may be used in conjunction with the putEvent() method,
 * but it may be used independently if external prescaling is done.
 * Basically, this method is used to tell the ConsumerPipe instance
 * that it should update its internal state for keeping track of whether
 * it is ready for another event.  
 *
 * This extra work is needed to support our fair-share event serving model.
 * With fair-event-serving, we need a way for external entities to get
 * a realistic event rate for a consumer by calling the isReadyForEvent()
 * method independent of whether events actually end up being queued.
 */
void ConsumerPipe::wasConsidered(double currentTime)
{
  lastConsideredEventTime_ = currentTime;
  rateRequestCounter_->addSample(1.0, currentTime);
}

/**
 * Adds the specified event to this consumer pipe.
 */
void ConsumerPipe::putEvent(boost::shared_ptr< std::vector<char> > bufPtr)
{
  // add this event to the queue
  boost::mutex::scoped_lock scopedLockForEventQueue(eventQueueLock_);

  // add the event to our statistics for "queued" events
  double now = BaseCounter::getCurrentTime();
  double sizeInMB = static_cast<double>(bufPtr->size()) / 1048576.0;
  ltQueueSizeWhenQueuedCounter_->addSample(eventQueue_.size());
  stQueueSizeWhenQueuedCounter_->addSample(eventQueue_.size(), now);
  longTermQueuedCounter_->addSample(sizeInMB);
  shortTermQueuedCounter_->addSample(sizeInMB, now);

  eventQueue_.push_back(bufPtr);

  while (eventQueue_.size() > maxQueueSize_) {
    eventQueue_.pop_front();
  }
  // if a push mode consumer actually push the event out to SMProxyServer
  if(pushMode_) {
    bool success = pushEvent();
    // update the time of the most recent successful transaction
    if(!success) ++pushEventFailures_;
    else
    {
      lastEventRequestTime_ = time(NULL);
      ++events_;
      // add the event to our statistics for "served" events
      longTermServedCounter_->addSample(sizeInMB);
      shortTermServedCounter_->addSample(sizeInMB, now);
    }
  }
}

/**
 * Fetches the next event from this consumer pipe.
 * If there are no events in the pipe, an empty shared_ptr will be returned
 * (ptr.get() == NULL).
 */
boost::shared_ptr< std::vector<char> > ConsumerPipe::getEvent()
{
  // 25-Aug-2005, KAB: clear out any stale event(s)
  if (isIdle() || isDisconnected())
  {
    this->clearQueue();
  }

  // fetch the most recent event
  boost::shared_ptr< std::vector<char> > bufPtr;
  {
    boost::mutex::scoped_lock scopedLockForEventQueue(eventQueueLock_);
    if (! eventQueue_.empty())
    {
      bufPtr = eventQueue_.front();
      eventQueue_.pop_front();

      // add the event to our statistics for "served" events
      double sizeInMB = static_cast<double>(bufPtr->size()) / 1048576.0;
      longTermServedCounter_->addSample(sizeInMB);
      shortTermServedCounter_->addSample(sizeInMB);
    }
  }

  // update the time of the most recent request
  lastEventRequestTime_ = time(NULL);

  // return the event
  return bufPtr;
}

bool ConsumerPipe::pushEvent()
{
  // fetch the most recent event
  boost::shared_ptr< std::vector<char> > bufPtr;
  {
    boost::mutex::scoped_lock scopedLockForEventQueue(eventQueueLock_);
    if (! eventQueue_.empty())
    {
      bufPtr = eventQueue_.front();
      eventQueue_.pop_front();
    }
  }
  if (bufPtr.get() == NULL)
  {
    edm::LogError("pushEvent") << "========================================";
    edm::LogError("pushEvent") << "pushEvent called with empty event queue!";
    return false;
  }

  // push the next event out to a push mode consumer (SMProxyServer)
  FDEBUG(5) << "pushing out event to " << consumerName_ << std::endl;
  stor::ReadData data;

  data.d_.clear();
  // check if curl handle was obtained (at ctor) if not try again
  if(han_==0)
  {
    han_ = curl_easy_init();
    if(han_==0)
    {
      edm::LogError("pushEvent") << "Could not create curl handle";
      return false;
    }
    headers_ = curl_slist_append(headers_, "Content-Type: application/octet-stream");
    headers_ = curl_slist_append(headers_, "Content-Transfer-Encoding: binary");
    // Avoid the Expect: 100 continue automatic header that gives a 2 sec delay
    headers_ = curl_slist_append(headers_, "Expect:");
    setopt(han_, CURLOPT_HTTPHEADER, headers_);
    setopt(han_, CURLOPT_URL, consumerName_.c_str());
    setopt(han_, CURLOPT_WRITEFUNCTION, func);
  }

  setopt(han_,CURLOPT_WRITEDATA,&data);

  // build the event message
  EventMsgView msgView(&(*bufPtr)[0]);

  // add the request message as a http post
  setopt(han_, CURLOPT_POSTFIELDS, msgView.startAddress());
  setopt(han_, CURLOPT_POSTFIELDSIZE, msgView.size());

  // send the HTTP POST, read the reply
  // explicitly close connection when using pthttp transport or sometimes it hangs
  // because somtimes curl does not see that the connection was closed and tries to reuse it
  setopt(han_,CURLOPT_FORBID_REUSE, 1);
  CURLcode messageStatus = curl_easy_perform(han_);

  if(messageStatus!=0)
  {
    cerr << "curl perform failed for pushEvent" << endl;
    edm::LogError("pushEvent") << "curl perform failed for pushEvent. "
        << "Could not register: probably XDAQ not running on Storage Manager"
        << " at " << consumerName_;
    return false;
  }
  // should really read the message to see if okay (if we had made one!)
  if(data.d_.length() == 0)
  {
    return true;
  } else {
    if(data.d_.length() > 0) {
      std::vector<char> buf(1024);
      int len = data.d_.length();
      buf.resize(len);
      for (int i=0; i<len ; i++) buf[i] = data.d_[i];
      const unsigned int MAX_DUMP_LENGTH = 1000;
      edm::LogError("pushEvent") << "========================================";
      edm::LogError("pushEvent") << "Unexpected pushEvent response!";
      if (data.d_.length() <= MAX_DUMP_LENGTH) {
        edm::LogError("pushEvent") << "Here is the raw text that was returned:";
        edm::LogError("pushEvent") << data.d_;
      }
      else {
        edm::LogError("pushEvent") << "Here are the first " << MAX_DUMP_LENGTH <<
          " characters of the raw text that was returned:";
        edm::LogError("pushEvent") << (data.d_.substr(0, MAX_DUMP_LENGTH));
      }
      edm::LogError("pushEvent") << "========================================";
    }
  }
  return false;
}

void ConsumerPipe::clearQueue()
{
  boost::mutex::scoped_lock scopedLockForEventQueue(eventQueueLock_);
  eventQueue_.clear();
}

std::vector<std::string> ConsumerPipe::getTriggerRequest() const
{
  return triggerSelection_;
}

void ConsumerPipe::setRegistryWarning(std::string const& message)
{
  // convert the string to a vector of char and then pass off the work
  std::vector<char> warningBuff(message.size());;
  std::copy(message.begin(), message.end(), warningBuff.begin());
  setRegistryWarning(warningBuff);
}

void ConsumerPipe::setRegistryWarning(std::vector<char> const& message)
{
  // assign the registry warning text before setting the warning flag
  // to avoid race conditions in which the hasRegistryWarning() method
  // would return true but the message hasn't been set (simpler than
  // adding a mutex for the warning message string)
  registryWarningMessage_ = message;
  registryWarningWasReported_ = true;
}

/**
 * Returns the number of events for the specified statistics types
 * (short term vs. long term; desired vs. queued vs. served).
 */
long long ConsumerPipe::getEventCount(STATS_TIME_FRAME timeFrame,
                                      STATS_SAMPLE_TYPE sampleType,
                                      double currentTime)
{
  if (timeFrame == SHORT_TERM) {
    if (sampleType == QUEUED_EVENTS) {
      return shortTermQueuedCounter_->getSampleCount(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return shortTermDesiredCounter_->getSampleCount(currentTime);
    }
    else {
      return shortTermServedCounter_->getSampleCount(currentTime);
    }
  }
  else {
    if (sampleType == QUEUED_EVENTS) {
      return longTermQueuedCounter_->getSampleCount();
    }
    else if (sampleType == DESIRED_EVENTS) {
      return longTermDesiredCounter_->getSampleCount();
    }
    else {
      return longTermServedCounter_->getSampleCount();
    }
  }
}

/**
 * Returns the rate of events for the specified statistics types
 * (short term vs. long term; desired vs. queued vs. served).
 */
double ConsumerPipe::getEventRate(STATS_TIME_FRAME timeFrame,
                                  STATS_SAMPLE_TYPE sampleType,
                                  double currentTime)
{
  if (timeFrame == SHORT_TERM) {
    if (sampleType == QUEUED_EVENTS) {
      return shortTermQueuedCounter_->getSampleRate(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return shortTermDesiredCounter_->getSampleRate(currentTime);
    }
    else {
      return shortTermServedCounter_->getSampleRate(currentTime);
    }
  }
  else {
    if (sampleType == QUEUED_EVENTS) {
      return longTermQueuedCounter_->getSampleRate(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return longTermDesiredCounter_->getSampleRate(currentTime);
    }
    else {
      return longTermServedCounter_->getSampleRate(currentTime);
    }
  }
}

/**
 * Returns the data rate for the specified statistics types
 * (short term vs. long term; desired vs. queued vs. served).
 */
double ConsumerPipe::getDataRate(STATS_TIME_FRAME timeFrame,
                                 STATS_SAMPLE_TYPE sampleType,
                                 double currentTime)
{
  if (timeFrame == SHORT_TERM) {
    if (sampleType == QUEUED_EVENTS) {
      return shortTermQueuedCounter_->getValueRate(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return shortTermDesiredCounter_->getValueRate(currentTime);
    }
    else {
      return shortTermServedCounter_->getValueRate(currentTime);
    }
  }
  else {
    if (sampleType == QUEUED_EVENTS) {
      return longTermQueuedCounter_->getValueRate(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return longTermDesiredCounter_->getValueRate(currentTime);
    }
    else {
      return longTermServedCounter_->getValueRate(currentTime);
    }
  }
}

/**
 * Returns the duration (in seconds) for the specified statistics types
 * (short term vs. long term; desired vs. queued vs. served).
 * "Duration" here means the length of time in which the specified
 * statistics have been collected.
 */
double ConsumerPipe::getDuration(STATS_TIME_FRAME timeFrame,
                                 STATS_SAMPLE_TYPE sampleType,
                                 double currentTime)
{
  if (timeFrame == SHORT_TERM) {
    if (sampleType == QUEUED_EVENTS) {
      return shortTermQueuedCounter_->getDuration(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return shortTermDesiredCounter_->getDuration(currentTime);
    }
    else {
      return shortTermServedCounter_->getDuration(currentTime);
    }
  }
  else {
    if (sampleType == QUEUED_EVENTS) {
      return longTermQueuedCounter_->getDuration(currentTime);
    }
    else if (sampleType == DESIRED_EVENTS) {
      return longTermDesiredCounter_->getDuration(currentTime);
    }
    else {
      return longTermServedCounter_->getDuration(currentTime);
    }
  }
}

/**
 * Returns the average queue size for the specified statistics types
 * (short term vs. long term; desired vs. queued).  The queue size
 * is sampled before any additional events are added.  For example,
 * the average queue size for "queued" events is the size before each
 * new event is queued.
 */
double ConsumerPipe::getAverageQueueSize(STATS_TIME_FRAME timeFrame,
                                         STATS_SAMPLE_TYPE sampleType,
                                         double currentTime)
{
  if (timeFrame == SHORT_TERM) {
    if (sampleType == QUEUED_EVENTS) {
      return stQueueSizeWhenQueuedCounter_->getValueAverage(currentTime);
    }
    else {
      return stQueueSizeWhenDesiredCounter_->getValueAverage(currentTime);
    }
  }
  else {
    if (sampleType == QUEUED_EVENTS) {
      return ltQueueSizeWhenQueuedCounter_->getValueAverage();
    }
    else {
      return ltQueueSizeWhenDesiredCounter_->getValueAverage();
    }
  }
}
