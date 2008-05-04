// $Id: DataProcessManager.cc,v 1.10 2008/04/16 16:43:13 biery Exp $

#include "EventFilter/SMProxyServer/interface/DataProcessManager.h"
#include "EventFilter/StorageManager/interface/SMCurlInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/ConsRegMessage.h"

#include "boost/bind.hpp"

#include "curl/curl.h"
#include <wait.h>

using namespace std;
using namespace edm;

using boost::thread;
using boost::bind;

namespace 
{
  const int voidptr_size = sizeof(void*);
}

namespace stor
{

  DataProcessManager::DataProcessManager():
    cmd_q_(edm::getEventBuffer(voidptr_size,50)),
    alreadyRegistered_(false),
    alreadyRegisteredDQM_(false),
    headerRefetchRequested_(false),
    buf_(2000),
    headerRetryInterval_(5),
    dqmServiceManager_(new stor::DQMServiceManager()),
    receivedEvents_(0),
    receivedDQMEvents_(0),
    samples_(100)
  {
    // for performance measurements
    pmeter_ = new stor::SMPerformanceMeter();
    init();
  } 

  DataProcessManager::~DataProcessManager()
  {
    delete pmeter_;
  }

  void DataProcessManager::init()
  {
    regpage_ =  "/registerConsumer";
    DQMregpage_ = "/registerDQMConsumer";
    eventpage_ = "/geteventdata";
    DQMeventpage_ = "/getDQMeventdata";
    headerpage_ = "/getregdata";
    consumerName_ = stor::PROXY_SERVER_NAME;
    //consumerPriority_ = "PushMode"; // this means push mode!
    consumerPriority_ = "Normal";
    DQMconsumerName_ = stor::PROXY_SERVER_NAME;
    //DQMconsumerPriority_ =  "PushMode"; // this means push mode!
    DQMconsumerPriority_ =  "Normal";

    const double MAX_REQUEST_INTERVAL = 300.0;  // seconds
    double maxEventRequestRate = 10.0; // just a default until set in config action
    if (maxEventRequestRate < (1.0 / MAX_REQUEST_INTERVAL)) {
      minEventRequestInterval_ = MAX_REQUEST_INTERVAL;
    }
    else {
      minEventRequestInterval_ = 1.0 / maxEventRequestRate;  // seconds
    }
    consumerId_ = (time(0) & 0xffffff);  // temporary - will get from ES later

    //double maxEventRequestRate = pset.getUntrackedParameter<double>("maxDQMEventRequestRate",1.0);
    maxEventRequestRate = 0.2; // TODO fixme: set this in the XML
    if (maxEventRequestRate < (1.0 / MAX_REQUEST_INTERVAL)) {
      minDQMEventRequestInterval_ = MAX_REQUEST_INTERVAL;
    }
    else {
      minDQMEventRequestInterval_ = 1.0 / maxEventRequestRate;  // seconds
    }
    DQMconsumerId_ = (time(0) & 0xffffff);  // temporary - will get from ES later

    alreadyRegistered_ = false;
    alreadyRegisteredDQM_ = false;
    headerRefetchRequested_ = false;

    edm::ParameterSet ps = ParameterSet();
    // TODO fixme: only request event types that are requested by connected consumers?

    // 16-Apr-2008, KAB: set maxEventRequestRate in the parameterSet that
    // we send to the storage manager now that we have the fair share
    // algorithm working in the SM.
    Entry maxRateEntry("maxEventRequestRate",
                       static_cast<double>(1.0 / minEventRequestInterval_),
                       false);
    ps.insert(true, "maxEventRequestRate", maxRateEntry);

    consumerPSetString_ = ps.toString();
    // TODO fixme: only request folders that connected consumers want?
    consumerTopFolderName_ = "*";
    //consumerTopFolderName_ = "C1";
    receivedEvents_ = 0;
    receivedDQMEvents_ = 0;
    pmeter_->init(samples_);
    stats_.fullReset();

    // initialize the counters that we use for statistics
    ltEventFetchTimeCounter_.reset(new ForeverCounter());
    stEventFetchTimeCounter_.reset(new RollingIntervalCounter(180,5,20));
    ltDQMFetchTimeCounter_.reset(new ForeverCounter());
    stDQMFetchTimeCounter_.reset(new RollingIntervalCounter(180,5,20));
  }

  void DataProcessManager::setMaxEventRequestRate(double rate)
  {
    const double MAX_REQUEST_INTERVAL = 300.0;  // seconds
    if(rate <= 0.0) return; // TODO make sure config is checked!
    if (rate < (1.0 / MAX_REQUEST_INTERVAL)) {
      minEventRequestInterval_ = MAX_REQUEST_INTERVAL;
    }
    else {
      minEventRequestInterval_ = 1.0 / rate;  // seconds
    }

    // 16-Apr-2008, KAB: set maxEventRequestRate in the parameterSet that
    // we send to the storage manager now that we have the fair share
    // algorithm working in the SM.
    edm::ParameterSet ps = ParameterSet();
    Entry maxRateEntry("maxEventRequestRate",
                       static_cast<double>(1.0 / minEventRequestInterval_),
                       false);
    ps.insert(true, "maxEventRequestRate", maxRateEntry);
    consumerPSetString_ = ps.toString();
  }

  void DataProcessManager::setMaxDQMEventRequestRate(double rate)
  {
    const double MAX_REQUEST_INTERVAL = 300.0;  // seconds
    if(rate <= 0.0) return; // TODO make sure config is checked!
    if (rate < (1.0 / MAX_REQUEST_INTERVAL)) {
      minDQMEventRequestInterval_ = MAX_REQUEST_INTERVAL;
    }
    else {
      minDQMEventRequestInterval_ = 1.0 / rate;  // seconds
    }
  }

  void DataProcessManager::run(DataProcessManager* t)
  {
    t->processCommands();
  }

  void DataProcessManager::start()
  {
    // called from a different thread to start things going

    me_.reset(new boost::thread(boost::bind(DataProcessManager::run,this)));
  }

  void DataProcessManager::stop()
  {
    // called from a different thread - trigger completion to the
    // data process manager loop

    edm::EventBuffer::ProducerBuffer cb(*cmd_q_);
    MsgCode mc(cb.buffer(),MsgCode::DONE);
    mc.setCode(MsgCode::DONE);
    cb.commit(mc.codeSize());
  }

  void DataProcessManager::join()
  {
    // invoked from a different thread - block until "me_" is done
    if(me_) me_->join();
  }

  void DataProcessManager::processCommands()
  {
    // called with this data process manager's own thread.
    // first register with the SM for each subfarm
    bool doneWithRegistration = false;
    // TODO fixme: improve method of hardcored fixed retries
    unsigned int count = 0; // keep of count of tries and quit after 255
    unsigned int maxcount = 255;
    bool doneWithDQMRegistration = false;
    unsigned int countDQM = 0; // keep of count of tries and quit after 255
    bool alreadysaid = false;
    bool alreadysaidDQM = false;

    bool gotOneHeader = false;
    unsigned int countINIT = 0; // keep of count of tries and quit after 255
    bool alreadysaidINIT = false;

    bool DoneWithJob = false;
    while(!DoneWithJob)
    {
      // work loop
      // if a header re-fetch has been requested, reset the header vars
      if (headerRefetchRequested_) {
        headerRefetchRequested_ = false;
        gotOneHeader = false;
        countINIT = 0;
      }
      // register as event consumer to all SM senders
      if(!alreadyRegistered_) {
        if(!doneWithRegistration)
        {
          waitBetweenRegTrys();
          bool success = registerWithAllSM();
          if(success) doneWithRegistration = true;
          ++count;
        }
        // TODO fixme: decide what to do after max tries
        if(count >= maxcount) edm::LogInfo("processCommands") << "Could not register with all SM Servers"
           << " after " << maxcount << " tries";
        if(doneWithRegistration && !alreadysaid) {
          edm::LogInfo("processCommands") << "Registered with all SM Event Servers";
          alreadysaid = true;
        }
        if(doneWithRegistration) alreadyRegistered_ = true;
      }
      // now register as DQM consumers
      if(!alreadyRegisteredDQM_) {
        if(!doneWithDQMRegistration)
        {
          waitBetweenRegTrys();
          bool success = registerWithAllDQMSM();
          if(success) doneWithDQMRegistration = true;
          ++countDQM;
        }
        // TODO fixme: decide what to do after max tries
        if(count >= maxcount) edm::LogInfo("processCommands") << "Could not register with all SM DQMEvent Servers"
          << " after " << maxcount << " tries";
        if(doneWithDQMRegistration && !alreadysaidDQM) {
          edm::LogInfo("processCommands") << "Registered with all SM DQMEvent Servers";
          alreadysaidDQM = true;
        }
        if(doneWithDQMRegistration) alreadyRegisteredDQM_ = true;
      }
      // now get one INIT header (product registry) and save it
      // as long as at least one SMsender registered with
      // TODO fixme: use the data member for got header to go across runs
      if(!gotOneHeader)
      {
        waitBetweenRegTrys();
        bool success = getAnyHeaderFromSM();
        if(success) gotOneHeader = true;
        ++countINIT;
      }
      if(countINIT >= maxcount) edm::LogInfo("processCommands") << "Could not get product registry!"
          << " after " << maxcount << " tries";
      if(gotOneHeader && !alreadysaidINIT) {
        edm::LogInfo("processCommands") << "Got the product registry";
        alreadysaidINIT = true;
      }
      if(alreadyRegistered_ && gotOneHeader && haveHeader()) {
        getEventFromAllSM();
      }
      if(alreadyRegisteredDQM_) {
        getDQMEventFromAllSM();
      }

      // check for any commands - empty() does not block
      if(!cmd_q_->empty())
      {
        // the next line blocks until there is an entry in cmd_q
        edm::EventBuffer::ConsumerBuffer cb(*cmd_q_);
        MsgCode mc(cb.buffer(),cb.size());

        if(mc.getCode()==MsgCode::DONE) DoneWithJob = true;
        // right now we will ignore all messages other than DONE
      }

    } // done with process loop   
    edm::LogInfo("processCommands") << "Received done - stopping";
    if(dqmServiceManager_.get() != NULL) dqmServiceManager_->stop();
  }

  void DataProcessManager::addSM2Register(std::string smURL)
  {
    // This smURL is the URN of the StorageManager without the page extension
    // Check if already in the list
    bool alreadyInList = false;
    if(smList_.size() > 0) {
       for(unsigned int i = 0; i < smList_.size(); ++i) {
         if(smURL.compare(smList_[i]) == 0) {
            alreadyInList = true;
            break;
         }
       }
    }
    if(alreadyInList) return;
    smList_.push_back(smURL);
    smRegMap_.insert(std::make_pair(smURL,0));
    struct timeval lastRequestTime;
    lastRequestTime.tv_sec = 0;
    lastRequestTime.tv_usec = 0;
    lastReqMap_.insert(std::make_pair(smURL,lastRequestTime));
  }

  void DataProcessManager::addDQMSM2Register(std::string DQMsmURL)
  {
    // Check if already in the list
    bool alreadyInList = false;
    if(DQMsmList_.size() > 0) {
       for(unsigned int i = 0; i < DQMsmList_.size(); ++i) {
         if(DQMsmURL.compare(DQMsmList_[i]) == 0) {
            alreadyInList = true;
            break;
         }
       }
    }
    if(alreadyInList) return;
    DQMsmList_.push_back(DQMsmURL);
    DQMsmRegMap_.insert(std::make_pair(DQMsmURL,0));
    struct timeval lastRequestTime;
    lastRequestTime.tv_sec = 0;
    lastRequestTime.tv_usec = 0;
    lastDQMReqMap_.insert(std::make_pair(DQMsmURL,lastRequestTime));
  }

  bool DataProcessManager::registerWithAllSM()
  {
    // One try at registering with the SM on each subfarm
    // return true if registered with all SM 
    // Only make one attempt and return so we can make this thread stop
    if(smList_.size() == 0) return false;
    bool allRegistered = true;
    for(unsigned int i = 0; i < smList_.size(); ++i) {
      if(smRegMap_[smList_[i] ] > 0) continue; // already registered
      int consumerid = registerWithSM(smList_[i]);
      if(consumerid > 0) smRegMap_[smList_[i] ] = consumerid;
      else allRegistered = false;
    }
    return allRegistered;
  }

  bool DataProcessManager::registerWithAllDQMSM()
  {
    // One try at registering with the SM on each subfarm
    // return true if registered with all SM 
    // Only make one attempt and return so we can make this thread stop
    if(DQMsmList_.size() == 0) return false;
    bool allRegistered = true;
    for(unsigned int i = 0; i < DQMsmList_.size(); ++i) {
      if(DQMsmRegMap_[DQMsmList_[i] ] > 0) continue; // already registered
      int consumerid = registerWithDQMSM(DQMsmList_[i]);
      if(consumerid > 0) DQMsmRegMap_[DQMsmList_[i] ] = consumerid;
      else allRegistered = false;
    }
    return allRegistered;
  }

  int DataProcessManager::registerWithSM(std::string smURL)
  {
    // Use this same registration method for both event data and DQM data
    // return with consumerID or 0 for failure
    stor::ReadData data;

    data.d_.clear();
    CURL* han = curl_easy_init();
    if(han==0)
    {
      edm::LogError("registerWithSM") << "Could not create curl handle";
      // this is a fatal error isn't it? Are we catching this? TODO
      throw cms::Exception("registerWithSM","DataProcessManager")
          << "Unable to create curl handle\n";
    }
    // set the standard http request options
    std::string url2use = smURL + regpage_;
    setopt(han,CURLOPT_URL,url2use.c_str());
    setopt(han,CURLOPT_WRITEFUNCTION,func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    // build the registration request message to send to the storage manager
    const int BUFFER_SIZE = 2000;
    char msgBuff[BUFFER_SIZE];
    ConsRegRequestBuilder requestMessage(msgBuff, BUFFER_SIZE, consumerName_,
                                         consumerPriority_, consumerPSetString_);

    // add the request message as a http post
    setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
    setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
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
      cerr << "curl perform failed for registration" << endl;
      edm::LogError("registerWithSM") << "curl perform failed for registration. "
        << "Could not register: probably XDAQ not running on Storage Manager"
        << " at " << smURL;
      return 0;
    }
    uint32 registrationStatus = ConsRegResponseBuilder::ES_NOT_READY;
    int consumerId = 0;
    if(data.d_.length() > 0)
    {
      int len = data.d_.length();
      FDEBUG(9) << "registerWithSM received len = " << len << std::endl;
      buf_.resize(len);
      for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

      try {
        ConsRegResponseView respView(&buf_[0]);
        registrationStatus = respView.getStatus();
        consumerId = respView.getConsumerId();
        if (eventServer_.get() != NULL) {
          eventServer_->setStreamSelectionTable(respView.getStreamSelectionTable());
        }
      }
      catch (cms::Exception excpt) {
        const unsigned int MAX_DUMP_LENGTH = 1000;
        edm::LogError("registerWithSM") << "========================================";
        edm::LogError("registerWithSM") << "Exception decoding the registerWithSM response!";
        if (data.d_.length() <= MAX_DUMP_LENGTH) {
          edm::LogError("registerWithSM") << "Here is the raw text that was returned:";
          edm::LogError("registerWithSM") << data.d_;
        }
        else {
          edm::LogError("registerWithSM") << "Here are the first " << MAX_DUMP_LENGTH <<
            " characters of the raw text that was returned:";
          edm::LogError("registerWithSM") << (data.d_.substr(0, MAX_DUMP_LENGTH));
        }
        edm::LogError("registerWithSM") << "========================================";
        return 0;
      }
    }
    if(registrationStatus == ConsRegResponseBuilder::ES_NOT_READY) return 0;
    FDEBUG(5) << "Consumer ID = " << consumerId << endl;
    return consumerId;
  }

  int DataProcessManager::registerWithDQMSM(std::string smURL)
  {
    // Use this same registration method for both event data and DQM data
    // return with consumerID or 0 for failure
    stor::ReadData data;

    data.d_.clear();
    CURL* han = curl_easy_init();
    if(han==0)
    {
      edm::LogError("registerWithDQMSM") << "Could not create curl handle";
      // this is a fatal error isn't it? Are we catching this? TODO
      throw cms::Exception("registerWithDQMSM","DataProcessManager")
          << "Unable to create curl handle\n";
    }
    // set the standard http request options
    std::string url2use = smURL + DQMregpage_;
    setopt(han,CURLOPT_URL,url2use.c_str());
    setopt(han,CURLOPT_WRITEFUNCTION,func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    // build the registration request message to send to the storage manager
    const int BUFFER_SIZE = 2000;
    char msgBuff[BUFFER_SIZE];
    ConsRegRequestBuilder requestMessage(msgBuff, BUFFER_SIZE, DQMconsumerName_,
                                         DQMconsumerPriority_, consumerTopFolderName_);

    // add the request message as a http post
    setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
    setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
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
      cerr << "curl perform failed for DQM registration" << endl;
      edm::LogError("registerWithDQMSM") << "curl perform failed for registration. "
        << "Could not register with DQM: probably XDAQ not running on Storage Manager"
        << " at " << smURL;
      return 0;
    }
    uint32 registrationStatus = ConsRegResponseBuilder::ES_NOT_READY;
    int consumerId = 0;
    if(data.d_.length() > 0)
    {
      int len = data.d_.length();
      FDEBUG(9) << "registerWithDQMSM received len = " << len << std::endl;
      buf_.resize(len);
      for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

      try {
        ConsRegResponseView respView(&buf_[0]);
        registrationStatus = respView.getStatus();
        consumerId = respView.getConsumerId();
      }
      catch (cms::Exception excpt) {
        const unsigned int MAX_DUMP_LENGTH = 1000;
        edm::LogError("registerWithDQMSM") << "========================================";
        edm::LogError("registerWithDQMSM") << "Exception decoding the registerWithSM response!";
        if (data.d_.length() <= MAX_DUMP_LENGTH) {
          edm::LogError("registerWithDQMSM") << "Here is the raw text that was returned:";
          edm::LogError("registerWithDQMSM") << data.d_;
        }
        else {
          edm::LogError("registerWithDQMSM") << "Here are the first " << MAX_DUMP_LENGTH <<
            " characters of the raw text that was returned:";
          edm::LogError("registerWithDQMSM") << (data.d_.substr(0, MAX_DUMP_LENGTH));
        }
        edm::LogError("registerWithDQMSM") << "========================================";
        return 0;
      }
    }
    if(registrationStatus == ConsRegResponseBuilder::ES_NOT_READY) return 0;
    FDEBUG(5) << "Consumer ID = " << consumerId << endl;
    return consumerId;
  }

  bool DataProcessManager::getAnyHeaderFromSM()
  {
    // Try the list of SM in order of registration to get one Header
    bool gotOneHeader = false;
    if(smList_.size() > 0) {
       for(unsigned int i = 0; i < smList_.size(); ++i) {
         if(smRegMap_[smList_[i] ] > 0) {
            bool success = getHeaderFromSM(smList_[i]);
            if(success) { // should cleam this up!
              gotOneHeader = true;
              return gotOneHeader;
            }
         }
       }
    } else {
      // this is a problem (but maybe not with non-blocking processing loop)
      return false;
    }
    return gotOneHeader;
  }

  bool DataProcessManager::getHeaderFromSM(std::string smURL)
  {
    // One single try to get a header from this SM URL
    stor::ReadData data;

    data.d_.clear();
    CURL* han = curl_easy_init();
    if(han==0)
    {
      edm::LogError("getHeaderFromSM") << "Could not create curl handle";
      // this is a fatal error isn't it? Are we catching this? TODO
      throw cms::Exception("getHeaderFromSM","DataProcessManager")
          << "Unable to create curl handle\n";
    }
    // set the standard http request options
    std::string url2use = smURL + headerpage_;
    setopt(han,CURLOPT_URL,url2use.c_str());
    setopt(han,CURLOPT_WRITEFUNCTION,func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    // send our consumer ID as part of the header request
    char msgBuff[100];
    OtherMessageBuilder requestMessage(&msgBuff[0], Header::HEADER_REQUEST,
                                       sizeof(char_uint32));
    uint8 *bodyPtr = requestMessage.msgBody();
    convert(smRegMap_[smURL], bodyPtr);
    setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
    setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
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
      cerr << "curl perform failed for header" << endl;
      edm::LogError("getHeaderFromSM") << "curl perform failed for header. "
        << "Could not get header from an already registered Storage Manager"
        << " at " << smURL;
      return false;
    }
    if(data.d_.length() == 0)
    { 
      return false;
    }

    // rely on http transfer string of correct length!
    int len = data.d_.length();
    FDEBUG(9) << "getHeaderFromSM received registry len = " << len << std::endl;

    // check that we've received a valid INIT message
    // or a set of INIT messages.  Save everything that we receive.
    bool addedNewInitMsg = false;
    try
    {
      HeaderView hdrView(&data.d_[0]);
      if (hdrView.code() == Header::INIT)
      {
        InitMsgView initView(&data.d_[0]);
        if (initMsgCollection_->addIfUnique(initView))
        {
          addedNewInitMsg = true;
        }
      }
      else if (hdrView.code() == Header::INIT_SET)
      {
        OtherMessageView otherView(&data.d_[0]);
        bodyPtr = otherView.msgBody();
        uint32 fullSize = otherView.bodySize();
        while ((unsigned int) (bodyPtr-otherView.msgBody()) < fullSize)
        {
          InitMsgView initView(bodyPtr);
          if (initMsgCollection_->addIfUnique(initView))
          {
            addedNewInitMsg = true;
          }
          bodyPtr += initView.size();
        }
      }
      else
      {
        throw cms::Exception("getHeaderFromSM", "DataProcessManager");
      }
    }
    catch (cms::Exception excpt)
    {
      const unsigned int MAX_DUMP_LENGTH = 1000;
      edm::LogError("getHeaderFromSM") << "========================================";
      edm::LogError("getHeaderFromSM") << "Exception decoding the getRegistryData response!";
      if (data.d_.length() <= MAX_DUMP_LENGTH) {
        edm::LogError("getHeaderFromSM") << "Here is the raw text that was returned:";
        edm::LogError("getHeaderFromSM") << data.d_;
      }
      else {
        edm::LogError("getHeaderFromSM") << "Here are the first " << MAX_DUMP_LENGTH <<
          " characters of the raw text that was returned:";
        edm::LogError("getHeaderFromSM") << (data.d_.substr(0, MAX_DUMP_LENGTH));
      }
      edm::LogError("getHeaderFromSM") << "========================================";
      throw excpt;
    }

    // check if any currently connected consumers did not specify
    // an HLT output module label and we now have multiple, different,
    // INIT messages.  If so, we need to complain because the
    // SelectHLTOutput parameter needs to be specified when there
    // is more than one HLT output module (and correspondingly, more
    // than one INIT message)
    if (addedNewInitMsg && eventServer_.get() != NULL &&
        initMsgCollection_->size() > 1)
    {
      std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable = 
        eventServer_->getConsumerTable();
      std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator 
        consumerIter;
      for (consumerIter = consumerTable.begin();
           consumerIter != consumerTable.end();
           consumerIter++)
      {
        boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;

        if (consPtr->getHLTOutputSelection().empty())
        {
          // store a warning message in the consumer pipe to be
          // sent to the consumer at the next opportunity
          std::string errorString;
          errorString.append("ERROR: The configuration for this ");
          errorString.append("consumer does not specify an HLT output ");
          errorString.append("module.\nPlease specify one of the HLT ");
          errorString.append("output modules listed below as the ");
          errorString.append("SelectHLTOutput parameter ");
          errorString.append("in the InputSource configuration.\n");
          errorString.append(initMsgCollection_->getSelectionHelpString());
          errorString.append("\n");
          consPtr->setRegistryWarning(errorString);
        }
      }
    }

    return true;
  }

  void DataProcessManager::waitBetweenRegTrys()
  {
    // for now just a simple wait for a fixed time
    sleep(headerRetryInterval_);
    return;
  }

  bool DataProcessManager::haveRegWithEventServer()
  {
    // registered with any of the SM event servers
    if(smList_.size() > 0) {
      for(unsigned int i = 0; i < smList_.size(); ++i) {
        if(smRegMap_[smList_[i] ] > 0) return true;
      }
    }
    return false;
  }

  bool DataProcessManager::haveRegWithDQMServer()
  {
    // registered with any of the SM DQM servers
    if(DQMsmList_.size() > 0) {
      for(unsigned int i = 0; i < DQMsmList_.size(); ++i) {
        if(DQMsmRegMap_[DQMsmList_[i] ] > 0) return true;
      }
    }
    return false;
  }

  bool DataProcessManager::haveHeader()
  {
    if(initMsgCollection_->size() > 0) return true;
    return false;
  }

  void DataProcessManager::getEventFromAllSM()
  {
    // Try the list of SM in order of registration to get one event
    // so long as we have the header from SM already
    if(smList_.size() > 0 && haveHeader()) {
      double time2wait = 0.0;
      double sleepTime = 300.0;
      bool gotOneEvent = false;
      bool gotOne = false;
      for(unsigned int i = 0; i < smList_.size(); ++i) {
        if(smRegMap_[smList_[i] ] > 0) {   // is registered
          gotOne = getOneEventFromSM(smList_[i], time2wait);
          if(gotOne) {
            gotOneEvent = true;
          } else {
            if(time2wait < sleepTime && time2wait >= 0.0) sleepTime = time2wait;
          }
        }
      }
      // check if we need to sleep (to enforce the allowed request rate)
      // we don't want to ping the StorageManager app too often
      if(!gotOneEvent) {
        if(sleepTime > 0.0) usleep(static_cast<int>(1000000 * sleepTime));
      }
    }
  }

  double DataProcessManager::getTime2Wait(std::string smURL)
  {
    // calculate time since last ping of this SM in seconds
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeDiff = (double) now.tv_sec;
    timeDiff -= (double) lastReqMap_[smURL].tv_sec;
    timeDiff += ((double) now.tv_usec / 1000000.0);
    timeDiff -= ((double) lastReqMap_[smURL].tv_usec / 1000000.0);
    if (timeDiff < minEventRequestInterval_)
    {
      return (minEventRequestInterval_ - timeDiff);
    }
    else
    {
      return 0.0;
    }
  }

  void DataProcessManager::setTime2Now(std::string smURL)
  {
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    lastReqMap_[smURL] = now;
  }

  bool DataProcessManager::getOneEventFromSM(std::string smURL, double& time2wait)
  {
    // See if we will exceed the request rate, if so just return false
    // Return values: 
    //    true = we have an event; false = no event (whatever reason)
    // time2wait values:
    //    0.0 = we pinged this SM this time; >0 = did not ping, wait this time
    // if every SM returns false we sleep some time
    time2wait = getTime2Wait(smURL);
    if(time2wait > 0.0) {
      return false;
    } else {
      setTime2Now(smURL);
    }

    // One single try to get a event from this SM URL
    stor::ReadData data;

    // start a measurement of how long the HTTP POST takes
    eventFetchTimer_.stop();
    eventFetchTimer_.reset();
    eventFetchTimer_.start();

    data.d_.clear();
    CURL* han = curl_easy_init();
    if(han==0)
    {
      edm::LogError("getOneEventFromSM") << "Could not create curl handle";
      // this is a fatal error isn't it? Are we catching this? TODO
      throw cms::Exception("getOneEventFromSM","DataProcessManager")
          << "Unable to create curl handle\n";
    }
    // set the standard http request options
    std::string url2use = smURL + eventpage_;
    setopt(han,CURLOPT_URL,url2use.c_str());
    setopt(han,CURLOPT_WRITEFUNCTION,stor::func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    // send our consumer ID as part of the event request
    // The event request body consists of the consumerId and the
    // number of INIT messages in our collection.  The latter is used
    // to determine if we need to re-fetch the INIT message collection.
    char msgBuff[100];
    OtherMessageBuilder requestMessage(&msgBuff[0], Header::EVENT_REQUEST,
                                       2 * sizeof(char_uint32));
    uint8 *bodyPtr = requestMessage.msgBody();
    convert(smRegMap_[smURL], bodyPtr);
    bodyPtr += sizeof(char_uint32);
    convert((uint32) initMsgCollection_->size(), bodyPtr);
    setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
    setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
    struct curl_slist *headers=NULL;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    headers = curl_slist_append(headers, "Content-Transfer-Encoding: binary");
    stor::setopt(han, CURLOPT_HTTPHEADER, headers);

    // send the HTTP POST, read the reply, and cleanup before going on
    CURLcode messageStatus = curl_easy_perform(han);
    curl_slist_free_all(headers);
    curl_easy_cleanup(han);

    if(messageStatus!=0)
    { 
      cerr << "curl perform failed for event" << endl;
      edm::LogError("getOneEventFromSM") << "curl perform failed for event. "
        << "Could not get event from an already registered Storage Manager"
        << " at " << smURL;

      // keep statistics for all HTTP POSTS
      eventFetchTimer_.stop();
      ltEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
      stEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

      return false;
    }

    // rely on http transfer string of correct length!
    int len = data.d_.length();
    FDEBUG(9) << "getOneEventFromSM received len = " << len << std::endl;
    if(data.d_.length() == 0)
    { 
      // keep statistics for all HTTP POSTS
      eventFetchTimer_.stop();
      ltEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
      stEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

      return false;
    }

    buf_.resize(len);
    for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

    // keep statistics for all HTTP POSTS
    eventFetchTimer_.stop();
    ltEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
    stEventFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

    // first check if done message
    OtherMessageView msgView(&buf_[0]);

    if (msgView.code() == Header::DONE) {
      // TODO fixme:just print message for now
      std::cout << " SM " << smURL << " has halted" << std::endl;
      return false;
    } else if (msgView.code() == Header::NEW_INIT_AVAILABLE) {
      std::cout << "Received NEW_INIT_AVAILABLE message" << std::endl;
      headerRefetchRequested_ = true;
      return false;
    } else {
      // 05-Feb-2008, KAB:  catch (and rethrow) any exceptions decoding
      // the event data so that we can display the returned HTML and
      // (hopefully) give the user a hint as to the cause of the problem.
      try {
        HeaderView hdrView(&buf_[0]);
        if (hdrView.code() != Header::EVENT) {
          throw cms::Exception("getOneEventFromSM", "DataProcessManager");
        }
        EventMsgView eventView(&buf_[0]);
        ++receivedEvents_;
        addMeasurement((unsigned long)data.d_.length());
        if(eventServer_.get() != NULL) {
          eventServer_->processEvent(eventView);
          return true;
        }
      }
      catch (cms::Exception excpt) {
        const unsigned int MAX_DUMP_LENGTH = 1000;
        edm::LogError("getOneEventFromSM") << "========================================";
        edm::LogError("getOneEventFromSM") << "Exception decoding the getEventData response!";
        if (data.d_.length() <= MAX_DUMP_LENGTH) {
          edm::LogError("getOneEventFromSM") << "Here is the raw text that was returned:";
          edm::LogError("getOneEventFromSM") << data.d_;
        }
        else {
          edm::LogError("getOneEventFromSM") << "Here are the first " << MAX_DUMP_LENGTH <<
            " characters of the raw text that was returned:";
          edm::LogError("getOneEventFromSM") << (data.d_.substr(0, MAX_DUMP_LENGTH));
        }
        edm::LogError("getOneEventFromSM") << "========================================";
        throw excpt;
      }
    }
    return false;
  }

  void DataProcessManager::getDQMEventFromAllSM()
  {
    // Try the list of SM in order of registration to get one event
    // so long as we have the header from SM already
    if(smList_.size() > 0) {
      double time2wait = 0.0;
      double sleepTime = 300.0;
      bool gotOneEvent = false;
      bool gotOne = false;
      for(unsigned int i = 0; i < smList_.size(); ++i) {
        if(DQMsmRegMap_[smList_[i] ] > 0) {   // is registered
          gotOne = getOneDQMEventFromSM(smList_[i], time2wait);
          if(gotOne) {
            gotOneEvent = true;
          } else {
            if(time2wait < sleepTime && time2wait >= 0.0) sleepTime = time2wait;
          }
        }
      }
      // check if we need to sleep (to enforce the allowed request rate)
      // we don't want to ping the StorageManager app too often
      // TODO fixme: Cannot sleep for DQM as this is a long time usually
      //             and we block the event request poll if we sleep!
      //             have to find out how to ensure the correct poll rate
      if(!gotOneEvent) {
        //if(sleepTime > 0.0) usleep(static_cast<int>(1000000 * sleepTime));
      }
    }
  }

  double DataProcessManager::getDQMTime2Wait(std::string smURL)
  {
    // calculate time since last ping of this SM in seconds
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeDiff = (double) now.tv_sec;
    timeDiff -= (double) lastDQMReqMap_[smURL].tv_sec;
    timeDiff += ((double) now.tv_usec / 1000000.0);
    timeDiff -= ((double) lastDQMReqMap_[smURL].tv_usec / 1000000.0);
    if (timeDiff < minDQMEventRequestInterval_)
    {
      return (minDQMEventRequestInterval_ - timeDiff);
    }
    else
    {
      return 0.0;
    }
  }

  void DataProcessManager::setDQMTime2Now(std::string smURL)
  {
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    lastDQMReqMap_[smURL] = now;
  }

  bool DataProcessManager::getOneDQMEventFromSM(std::string smURL, double& time2wait)
  {
    // See if we will exceed the request rate, if so just return false
    // Return values: 
    //    true = we have an event; false = no event (whatever reason)
    // time2wait values:
    //    0.0 = we pinged this SM this time; >0 = did not ping, wait this time
    // if every SM returns false we sleep some time
    time2wait = getDQMTime2Wait(smURL);
    if(time2wait > 0.0) {
      return false;
    } else {
      setDQMTime2Now(smURL);
    }

    // One single try to get a event from this SM URL
    stor::ReadData data;

    // start a measurement of how long the HTTP POST takes
    dqmFetchTimer_.stop();
    dqmFetchTimer_.reset();
    dqmFetchTimer_.start();

    data.d_.clear();
    CURL* han = curl_easy_init();
    if(han==0)
    {
      edm::LogError("getOneDQMEventFromSM") << "Could not create curl handle";
      // this is a fatal error isn't it? Are we catching this? TODO
      throw cms::Exception("getOneDQMEventFromSM","DataProcessManager")
          << "Unable to create curl handle\n";
    }
    // set the standard http request options
    std::string url2use = smURL + DQMeventpage_;
    setopt(han,CURLOPT_URL,url2use.c_str());
    setopt(han,CURLOPT_WRITEFUNCTION,stor::func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    // send our consumer ID as part of the event request
    char msgBuff[100];
    OtherMessageBuilder requestMessage(&msgBuff[0], Header::DQMEVENT_REQUEST,
                                       sizeof(char_uint32));
    uint8 *bodyPtr = requestMessage.msgBody();
    convert(DQMsmRegMap_[smURL], bodyPtr);
    setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
    setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
    struct curl_slist *headers=NULL;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    headers = curl_slist_append(headers, "Content-Transfer-Encoding: binary");
    stor::setopt(han, CURLOPT_HTTPHEADER, headers);

    // send the HTTP POST, read the reply, and cleanup before going on
    CURLcode messageStatus = curl_easy_perform(han);
    curl_slist_free_all(headers);
    curl_easy_cleanup(han);

    if(messageStatus!=0)
    { 
      cerr << "curl perform failed for DQM event" << endl;
      edm::LogError("getOneDQMEventFromSM") << "curl perform failed for DQM event. "
        << "Could not get DQMevent from an already registered Storage Manager"
        << " at " << smURL;

      // keep statistics for all HTTP POSTS
      dqmFetchTimer_.stop();
      ltDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
      stDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

      return false;
    }

    // rely on http transfer string of correct length!
    int len = data.d_.length();
    FDEBUG(9) << "getOneDQMEventFromSM received len = " << len << std::endl;
    if(data.d_.length() == 0)
    { 
      // keep statistics for all HTTP POSTS
      dqmFetchTimer_.stop();
      ltDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
      stDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

      return false;
    }

    buf_.resize(len);
    for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

    // keep statistics for all HTTP POSTS
    dqmFetchTimer_.stop();
    ltDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());
    stDQMFetchTimeCounter_->addSample(eventFetchTimer_.realTime());

    // first check if done message
    OtherMessageView msgView(&buf_[0]);

    if (msgView.code() == Header::DONE) {
      // TODO fixme:just print message for now
      std::cout << " SM " << smURL << " has halted" << std::endl;
      return false;
    } else {
      DQMEventMsgView dqmEventView(&buf_[0]);
      ++receivedDQMEvents_;
      addMeasurement((unsigned long)data.d_.length());
      if(dqmServiceManager_.get() != NULL) {
          dqmServiceManager_->manageDQMEventMsg(dqmEventView);
          return true;
      }
    }
    return false;
  }

//////////// ***  Performance //////////////////////////////////////////////////////////
  void DataProcessManager::addMeasurement(unsigned long size)
  {
    // for bandwidth performance measurements
    if(pmeter_->addSample(size))
    {
       stats_ = pmeter_->getStats();
    }
  }

  double DataProcessManager::getSampleCount(STATS_TIME_FRAME timeFrame,
                                            STATS_TIMING_TYPE timingType,
                                            double currentTime)
  {
    if (timeFrame == SHORT_TERM) {
      if (timingType == DQMEVENT_FETCH) {
        return stDQMFetchTimeCounter_->getSampleCount(currentTime);
      }
      else {
        return stEventFetchTimeCounter_->getSampleCount(currentTime);
      }
    }
    else {
      if (timingType == DQMEVENT_FETCH) {
        return ltDQMFetchTimeCounter_->getSampleCount();
      }
      else {
        return ltEventFetchTimeCounter_->getSampleCount();
      }
    }
  }

  double DataProcessManager::getAverageValue(STATS_TIME_FRAME timeFrame,
                                             STATS_TIMING_TYPE timingType,
                                             double currentTime)
  {
    if (timeFrame == SHORT_TERM) {
      if (timingType == DQMEVENT_FETCH) {
        return stDQMFetchTimeCounter_->getValueAverage(currentTime);
      }
      else {
        return stEventFetchTimeCounter_->getValueAverage(currentTime);
      }
    }
    else {
      if (timingType == DQMEVENT_FETCH) {
        return ltDQMFetchTimeCounter_->getValueAverage();
      }
      else {
        return ltEventFetchTimeCounter_->getValueAverage();
      }
    }
  }

  double DataProcessManager::getDuration(STATS_TIME_FRAME timeFrame,
                                         STATS_TIMING_TYPE timingType,
                                         double currentTime)
  {
    if (timeFrame == SHORT_TERM) {
      if (timingType == DQMEVENT_FETCH) {
        return stDQMFetchTimeCounter_->getDuration(currentTime);
      }
      else {
        return stEventFetchTimeCounter_->getDuration(currentTime);
      }
    }
    else {
      if (timingType == DQMEVENT_FETCH) {
        return ltDQMFetchTimeCounter_->getDuration(currentTime);
      }
      else {
        return ltEventFetchTimeCounter_->getDuration(currentTime);
      }
    }
  }
}
