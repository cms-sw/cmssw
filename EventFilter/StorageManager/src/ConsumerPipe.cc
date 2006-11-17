/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

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
 * ConsumerPipe constructor.
 */
ConsumerPipe::ConsumerPipe(std::string name, std::string priority,
                           int activeTimeout, int idleTimeout,
                           boost::shared_ptr<edm::ParameterSet> parameterSet):
  consumerName_(name),consumerPriority_(priority),
  requestParamSet_(parameterSet)
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
 * Initializes the event selection for this consumer based on the
 * list of available triggers stored in the specified InitMsgView
 * and the request ParameterSet that was specified in the constructor.
 */
void ConsumerPipe::initializeSelection(InitMsgView const& initView)
{
  FDEBUG(5) << "Initializing consumer pipe, ID = " <<
    consumerId_ << std::endl;

  // TODO: fetch the list of trigger names from the init message
  //std::vector<std::string> triggerNameList;
  //triggerNameList.push_back("kab1");
  //triggerNameList.push_back("kab2");
  //triggerNameList.push_back("kab3");
  Strings triggerNameList;
  initView.hltTriggerNames(triggerNameList);

  // fake the process name (not yet available from the init message?)
  std::string processName = "HLT";

  /* ---printout the trigger names in the INIT message
  std::cout << ">>>>>>>>>>>Trigger names:" << std::endl;
  for(int i=0; i< triggerNameList.size(); ++i)
    std::cout<< ">>>>>>>>>>>  name = " << triggerNameList[i] << std::endl;
  */
  // create our event selector
  eventSelector_.reset(new EventSelector(requestParamSet_->getUntrackedParameter("SelectEvents", ParameterSet()),
					 triggerNameList));
  // indicate that initialization is complete
  initializationDone = true;

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
  // 13-Oct-2006, KAB - we're not ready if we haven't been initialized
  if (! initializationDone) return false;

  // for now, just test if we are in the active state
  time_t timeDiff = time(NULL) - lastEventRequestTime_;
  return (timeDiff < timeToIdleState_);
}

/**
 * Tests if the consumer wants the specified event.
 */
bool ConsumerPipe::wantsEvent(EventMsgView const& eventView) const
{
  // for now, take every event
  // TODO - start using eventSelector_
  std::vector<unsigned char> hlt_out;
  hlt_out.resize(1 + (eventView.hltCount()-1)/4);
  eventView.hltTriggerBits(&hlt_out[0]);
  /* --- print the trigger bits from the event header
  std::cout << ">>>>>>>>>>>Trigger bits:" << std::endl;
  for(int i=0; i< hlt_out.size(); ++i)
  {
    unsigned test = (unsigned int)hlt_out[i];
    std::cout<< hex << ">>>>>>>>>>>  bits = " << test << " " << hlt_out[i] << std::endl;
  }
  cout << "\nhlt bits=\n(";
  for(int i=(hlt_out.size()-1); i != -1 ; --i) 
     printBits(hlt_out[i]);
  cout << ")\n";
  */
  int num_paths = eventView.hltCount();
  bool rc = (eventSelector_->wantAll() || eventSelector_->acceptEvent(&hlt_out[0], num_paths));
  //std::cout << "====================== " << std::endl;
  //std::cout << "return selector code = " << rc << std::endl;
  //std::cout << "====================== " << std::endl;
  //return true;
  return rc;
}

/**
 * Adds the specified event to this consumer pipe.
 */
void ConsumerPipe::putEvent(boost::shared_ptr< std::vector<char> > bufPtr)
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
boost::shared_ptr< std::vector<char> > ConsumerPipe::getEvent()
{
  // 25-Aug-2005, KAB: clear out any stale event(s)
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
