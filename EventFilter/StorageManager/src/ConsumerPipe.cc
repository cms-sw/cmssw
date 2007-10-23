/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/ConsumerPipe.h"
#include "EventFilter/StorageManager/interface/SMCurlInterface.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "curl/curl.h"

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
  requestParamSet_(parameterSet),
  pushEventFailures_(0)
{
  // initialize the time values we use for defining "states"
  timeToIdleState_ = activeTimeout;
  timeToDisconnectedState_ = activeTimeout + idleTimeout;
  lastEventRequestTime_ = time(NULL);
  initializationDone = false;
  pushMode_ = false;
  if(consumerPriority_.compare("SMProxyServer") == 0) pushMode_ = true;

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

  // fetch the list of trigger names from the init message
  Strings triggerNameList;
  initView.hltTriggerNames(triggerNameList);

  // TODO fake the process name (not yet available from the init message?)
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
  // get trigger bits for this event and check using eventSelector_
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
  // if a push mode consumer actually push the event out to SMProxyServer
  if(pushMode_) {
    bool success = pushEvent();
    if(!success) ++pushEventFailures_;
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

bool ConsumerPipe::pushEvent()
{
  // push the next event out to a push mode consumer (SMProxyServer)
  FDEBUG(5) << "pushing out event to " << consumerName_ << std::endl;
  stor::ReadData data;

  data.d_.clear();
  CURL* han = curl_easy_init();
  if(han==0)
  {
    edm::LogError("pushEvent") << "Could not create curl handle";
    return false;
  }
  // set the standard http request options
  setopt(han,CURLOPT_URL,consumerName_.c_str());
  setopt(han,CURLOPT_WRITEFUNCTION,func);
  setopt(han,CURLOPT_WRITEDATA,&data);

  // build the event message
  EventMsgView msgView(&(*latestEvent_)[0]);

  // add the request message as a http post
  setopt(han, CURLOPT_POSTFIELDS, msgView.startAddress());
  setopt(han, CURLOPT_POSTFIELDSIZE, msgView.size());
  struct curl_slist *headers=NULL;
  headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
  headers = curl_slist_append(headers, "Content-Transfer-Encoding: binary");
  setopt(han, CURLOPT_HTTPHEADER, headers);

  // send the HTTP POST, read the reply, and cleanup before going on
  CURLcode messageStatus = curl_easy_perform(han);
  curl_slist_free_all(headers);
  curl_easy_cleanup(han);

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
  boost::mutex::scoped_lock scopedLockForLatestEvent(latestEventLock_);
  latestEvent_.reset();
}
