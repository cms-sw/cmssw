/**
 * This class is used to manage the subscriptions, DQMevents, and
 * lost connections associated with a DQMevent consumer within the
 * DQMevent server part of the storage manager or SM Proxy Server.
 *
 * Initial Implementation based on Kurt's ConsumerPipe
 * make a common class later when all this works
 *
 * $Id$
 */

#include "EventFilter/StorageManager/interface/DQMConsumerPipe.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

// keep this for debugging
//#include "IOPool/Streamer/interface/DumpTools.h"

using namespace std;
using namespace stor;
using namespace edm;

/**
 * Initialize the static value for the root consumer id.
 */
uint32 DQMConsumerPipe::rootId_ = 0;

/**
 * Initialize the static lock used to control access to the root ID.
 */
boost::mutex DQMConsumerPipe::rootIdLock_;

/**
 * DQMConsumerPipe constructor.
 */
DQMConsumerPipe::DQMConsumerPipe(std::string name, std::string priority,
                           int activeTimeout, int idleTimeout,
                           std::string folderName):
  consumerName_(name),consumerPriority_(priority),
  topFolderName_(folderName)
{
  // initialize the time values we use for defining "states"
  timeToIdleState_ = activeTimeout;
  timeToDisconnectedState_ = activeTimeout + idleTimeout;
  lastEventRequestTime_ = time(NULL);
  initializationDone = false;

  // assign the consumer ID
  boost::mutex::scoped_lock scopedLockForRootId(rootIdLock_);
  consumerId_ = rootId_;
  rootId_++;
}

/**
 * DQMConsumerPipe destructor.
 */
DQMConsumerPipe::~DQMConsumerPipe()
{
  FDEBUG(5) << "Executing destructor for DQM consumer pipe with ID = " <<
    consumerId_ << std::endl;
}

/**
 * Returns the consumer ID associated with this pipe.
 */
uint32 DQMConsumerPipe::getConsumerId() const
{
  return consumerId_;
}

/**
 * Initializes the event selection for this consumer based on the
 * list of available triggers stored in the specified InitMsgView
 * and the request ParameterSet that was specified in the constructor.
 */
void DQMConsumerPipe::initializeSelection()
{
  FDEBUG(5) << "Initializing DQM consumer pipe, ID = " <<
    consumerId_ << std::endl;

  // no need for initialization yet
  initializationDone = true;

}

/**
 * Tests if the consumer is idle (as opposed to active).  The idle
 * state indicates that the consumer is still connected, but it hasn't
 * requested an event in some time.
 */
bool DQMConsumerPipe::isIdle() const
{
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff >= timeToIdleState_ &&
          timeDiff <  timeToDisconnectedState_);
}

/**
 * Tests if the consumer has disconnected.
 */
bool DQMConsumerPipe::isDisconnected() const
{
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff >= timeToDisconnectedState_);
}

/**
 * Tests if the consumer is ready for an event.
 */
bool DQMConsumerPipe::isReadyForEvent() const
{
  // 13-Oct-2006, KAB - we're not ready if we haven't been initialized
  if (! initializationDone) return false;

  // for now, just test if we are in the active state
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff < timeToIdleState_);
}

/**
 * Tests if the consumer wants the specified event.
 */
bool DQMConsumerPipe::wantsDQMEvent(DQMEventMsgView const& eventView) const
{
  // for now, only allow one top folder selection or "*"
  if(topFolderName_.compare("*") == 0) return true;
  else return (topFolderName_.compare(eventView.topFolderName()) == 0);
}

/**
 * Adds the specified event to this consumer pipe.
 */
void DQMConsumerPipe::putDQMEvent(boost::shared_ptr< std::vector<char> > bufPtr)
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
boost::shared_ptr< std::vector<char> > DQMConsumerPipe::getDQMEvent()
{
  // clear out any stale event(s)
  if (isIdle() || isDisconnected())
  {
    latestEvent_.reset();
  }

  // fetch the most recent event
  boost::shared_ptr< std::vector<char> > bufPtr;
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
