// $Id: DataProcessManager.cc,v 1.3 2007/05/16 22:57:45 hcheung Exp $

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

  DataProcessManager::~DataProcessManager()
  {
  }

  DataProcessManager::DataProcessManager():
    cmd_q_(edm::getEventBuffer(voidptr_size,50)),
    alreadyRegistered_(false),
    ser_prods_size_(0),
    serialized_prods_(1000000),
    buf_(2000),
    headerRetryInterval_(5),
    dqmServiceManager_(new stor::DQMServiceManager())
  {
    init();
  } 

  void DataProcessManager::init()
  {
    regpage_ =  "/registerConsumer";
    DQMregpage_ = "/registerDQMConsumer";
    headerpage_ = "/getregdata";
    consumerName_ = "Unknown";
    consumerPriority_ = "SMProxyServer";
    DQMconsumerName_ = "Unknown";
    DQMconsumerPriority_ =  "SMProxyServer";

    alreadyRegistered_ = false;

    edm::ParameterSet ps = ParameterSet();
    consumerPSetString_ = ps.toString();
    consumerTopFolderName_ = "*";
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
    // TODO to use non-blocking command queue method to process
    // stop commands while in registration loop: possible with tag in 15x series
    // using hack where we do not register again after a stopAction()
    if(!alreadyRegistered_) {
      bool doneWithRegistration = false;
      unsigned int count = 0; // keep of count of tries and quite after 5
      while(!doneWithRegistration && (count < 5))
      {
        bool success = registerWithAllSM();
        if(success) doneWithRegistration = true;
        else waitBetweenRegTrys();
        ++count;
      }
      if(count >= 5) edm::LogInfo("processCommands") << "Could not register with all SM Servers";
      else edm::LogInfo("processCommands") << "Registered with all SM Event Servers";
      // now register as DQM consumers
      doneWithRegistration = false;
      count = 0;
      while(!doneWithRegistration && (count < 5))
      {
        bool success = registerWithAllDQMSM();
        if(success) doneWithRegistration = true;
        else waitBetweenRegTrys();
        ++count;
      }
      if(count >= 5) edm::LogInfo("processCommands") << "Could not register with all SM DQMEvent Servers";
      else edm::LogInfo("processCommands") << "Registered with all SM DQMEvent Servers";
      // now get one INIT header (product registry) and save it
      bool gotOneHeader = false;
      count = 0;
      while(!gotOneHeader && (count < 5))
      {
        bool success = getAnyHeaderFromSM();
        if(success) gotOneHeader = true;
        else waitBetweenRegTrys();
        ++count;
      }
      if(count >= 5) edm::LogInfo("processCommands") << "Could not get product registry!";
      else edm::LogInfo("processCommands") << "Got the product registry";

      alreadyRegistered_ = true;
    } else {
      edm::LogInfo("processCommands") << "Reusing SM registration from previous run";
    }

    // just wait for command messages now
    while(1)
    {
      // check for any commands - empty() does not block
      // the next line blocks until there is an entry in cmd_q
      edm::EventBuffer::ConsumerBuffer cb(*cmd_q_);
      MsgCode mc(cb.buffer(),cb.size());

      if(mc.getCode()==MsgCode::DONE) break;
      // right now we will ignore all other messages
    }
    /* --- uncomment this bit instead to process commands without block
          // then we can stop while registering (possible in 15x series only)
    bool DoneWithJob = false;
    while(!DoneWithJob)
    {
      // work loop
      sleep(1);

      // check for any commands - empty() does not block
      if(!cmd_q_->empty())
      {
        // the next line blocks until there is an entry in cmd_q
        edm::EventBuffer::ConsumerBuffer cb(*cmd_q_);
        MsgCode mc(cb.buffer(),cb.size());

        if(mc.getCode()==MsgCode::DONE) DoneWithJob = true;
        // right now we will ignore all messages
      }

    }    
    */
    std::cout << "Received done - stopping" << std::endl;
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
    char_uint32 convertedId;
    convert(smRegMap_[smURL], convertedId);
    for (unsigned int idx = 0; idx < sizeof(char_uint32); idx++) {
      bodyPtr[idx] = convertedId[idx];
    }
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
    serialized_prods_.resize(len);
    for (int i=0; i<len ; i++) serialized_prods_[i] = data.d_[i];
    ser_prods_size_ = len;

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
    if(ser_prods_size_ > 0) return true;
    return false;
  }

  unsigned int DataProcessManager::headerSize()
  {
    return ser_prods_size_;
  }

  std::vector<unsigned char> DataProcessManager::getHeader()
  {
    return serialized_prods_;
  }
}
