/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

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
 * ConsumerPipe constructor.
 */
ConsumerPipe::ConsumerPipe(std::string name, std::string priority,
                           int activeTimeout, int idleTimeout):
  consumerName_(name),consumerPriority_(priority)
{
  // initialize the time values we use for defining "states"
  timeToIdleState_ = activeTimeout;
  timeToDisconnectedState_ = activeTimeout + idleTimeout;
  lastEventRequestTime_ = time(NULL);

  // assign the consumer ID
  boost::mutex::scoped_lock scopedLockForRootId(rootIdLock_);
  consumerId_ = rootId_;
  rootId_++;
}

/**
 * ConsumerPipe destructor.
 */
ConsumerPipe::~ConsumerPipe()
{
  FDEBUG(5) << "Executing destructor for consumer pipe with ID = " <<
    consumerId_ << std::endl;
}

/**
 * Returns the consumer ID associated with this pipe.
 */
uint32 ConsumerPipe::getConsumerId() const
{
  return consumerId_;
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
 * Tests if the consumer is ready for an event.
 */
bool ConsumerPipe::isReadyForEvent() const
{
  // for now, just test if we are in the active state
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff < timeToIdleState_);
}

/**
 * Tests if the consumer wants the specified event.
 */
bool ConsumerPipe::wantsEvent(const EventMsgView &eventView) const
{
  // for now, take every event
  return true;
}

/**
 * Adds the specified event to this consumer pipe.
 */
void ConsumerPipe::putEvent(boost::shared_ptr< vector<char> > bufPtr)
{
  // update the local pointer to the most recent event
  boost::mutex::scoped_lock scopedLockForLatestEvent(latestEventLock_);
  latestEvent_ = bufPtr;
}

/**
 * Fetches the next event from this consumer pipe.
 * If there are no events in the pipe, an empty shared_ptr will be returned
 * (ptr.get() == NULL).
 */
boost::shared_ptr< vector<char> > ConsumerPipe::getEvent()
{
  // 25-Aug-2005, KAB: clear out any stale event(s)
  if (isIdle() || isDisconnected())
  {
    latestEvent_.reset();
  }

  // fetch the most recent event
  boost::shared_ptr< vector<char> > bufPtr;
  {
    boost::mutex::scoped_lock scopedLockForLatestEvent(latestEventLock_);
    //bufPtr_ = latestEvent_;
    //latestEvent_.reset();
    bufPtr.swap(latestEvent_);
  }

  // update the time of the most recent request
  lastEventRequestTime_ = time(NULL);

  // return the event
  return bufPtr;
}
