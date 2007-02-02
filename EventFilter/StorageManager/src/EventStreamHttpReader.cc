/*
  Input source for event consumers that will get events from the
  Storage Manager Event Server. This does uses a HTTP get using the
  cURL library. The Storage Manager Event Server responses with
  a binary octet-stream.  The product registry is also obtained
  through a HTTP get.
      There is currently no test of the product registry against
  the consumer client product registry within the code. It should
  already be done if this was inherenting from the standard
  framework input source. Currently we inherit from InputSource.

  17 Mar 2006 - HWKC - initial version for testing
  30 Mar 2006 - HWKC - first proof of principle version that can
                wait for the the product registry and events from
                the Storage Manager. Only way to stop the cmsRun
                using this input source is to kill the Storage
                Manager or specify a maximum number of events for
                the client to read through a maxEvents parameter.
*/

#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "EventFilter/StorageManager/interface/ConsumerPipe.h"

#include <algorithm>
#include <iterator>
#include "curl/curl.h"
#include <string>

#include <wait.h>

using namespace std;
using namespace edm;

namespace edm
{  
  struct ReadData
  {
    std::string d_;
  };  

  size_t func(void* buf,size_t size, size_t nmemb, void* userp)
  {
    ReadData* rdata = (ReadData*)userp;
    size_t sz = size * nmemb;
    char* cbuf = (char*)buf;
    rdata->d_.insert(rdata->d_.end(),cbuf,cbuf+sz);
    return sz;
  }

  template <class Han, class Opt, class Par>
  int setopt(Han han,Opt opt,Par par)
  {
    if(curl_easy_setopt(han,opt,par)!=0)
      {
        cerr << "could not setopt " << opt << endl;
        abort();
      }
    return 0;
  }

  // ----------------------------------

  EventStreamHttpReader::EventStreamHttpReader(edm::ParameterSet const& ps,
					       edm::InputSourceDescription const& desc):
    edm::StreamerInputSource(ps, desc),
    sourceurl_(ps.getParameter<string>("sourceURL")),
    buf_(1000*1000*7), 
    events_read_(0)
  {
    std::string evturl = sourceurl_ + "/geteventdata";
    int stlen = evturl.length();
    for (int i=0; i<stlen; i++) eventurl_[i]=evturl[i];
    eventurl_[stlen] = '\0';

    std::string header = sourceurl_ + "/getregdata";
    stlen = header.length();
    for (int i=0; i<stlen; i++) headerurl_[i]=header[i];
    headerurl_[stlen] = '\0';

    std::string regurl = sourceurl_ + "/registerConsumer";
    stlen = regurl.length();
    for (int i=0; i<stlen; i++) subscriptionurl_[i]=regurl[i];
    subscriptionurl_[stlen] = '\0';

    // 09-Aug-2006, KAB: new parameters
    const double MAX_REQUEST_INTERVAL = 300.0;  // seconds
    consumerName_ = ps.getUntrackedParameter<string>("consumerName","Unknown");
    consumerPriority_ = ps.getUntrackedParameter<string>("consumerPriority","normal");
    headerRetryInterval_ = ps.getUntrackedParameter<int>("headerRetryInterval",5);
    double maxEventRequestRate = ps.getUntrackedParameter<double>("maxEventRequestRate",1.0);
    if (maxEventRequestRate < (1.0 / MAX_REQUEST_INTERVAL)) {
      minEventRequestInterval_ = MAX_REQUEST_INTERVAL;
    }
    else {
      minEventRequestInterval_ = 1.0 / maxEventRequestRate;  // seconds
    }
    lastRequestTime_.tv_sec = 0;
    lastRequestTime_.tv_usec = 0;

    // 28-Aug-2006, KAB: save our parameter set in string format to
    // be sent to the event server to specify our "request" (that is, which
    // events we are interested in).
    consumerPSetString_ = ps.toString();

    // 16-Aug-2006, KAB: register this consumer with the event server
    consumerId_ = (time(0) & 0xffffff);  // temporary - will get from ES later
    registerWithEventServer();

    std::auto_ptr<SendJobHeader> p = readHeader();
    SendDescs & descs = p->descs_;
    mergeWithRegistry(descs, productRegistry());

    // next taken from IOPool/Streamer/EventStreamFileReader
    // jbk - the next line should not be needed
    declareStreamers(descs);
    buildClassCache(descs);
    loadExtraClasses();
  }

  EventStreamHttpReader::~EventStreamHttpReader()
  {
  }

  std::auto_ptr<edm::EventPrincipal> EventStreamHttpReader::read()
  {
    // repeat a http get every 5 seconds until we get an event
    // wait for Storage Manager event server buffer to not be empty
    // only way to stop is specify a maxEvents parameter
    // or kill the STorage Manager so the http get fails.
    // do it like test for the proof of principle test

    // see if already read maxEvents
    if(maxEvents() > 0 && events_read_ >= maxEvents()) 
      return std::auto_ptr<edm::EventPrincipal>();

    // check if we need to sleep (to enforce the allowed request rate)
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeDiff = (double) now.tv_sec;
    timeDiff -= (double) lastRequestTime_.tv_sec;
    timeDiff += ((double) now.tv_usec / 1000000.0);
    timeDiff -= ((double) lastRequestTime_.tv_usec / 1000000.0);
    //cout << "timeDiff = " << timeDiff
    //     << ", minTime = " << minEventRequestInterval_ << std::endl;
    if (timeDiff < minEventRequestInterval_)
    {
      double sleepTime = minEventRequestInterval_ - timeDiff;
      // trim off a little sleep time to account for the time taken by
      // calling gettimeofday again
      sleepTime -= 0.01;
      if (sleepTime < 0.0) {sleepTime = 0.0;}
      //cout << "sleeping for " << sleepTime << endl;
      usleep(static_cast<int>(1000000 * sleepTime));
      gettimeofday(&lastRequestTime_, &dummyTZ);
    }
    else
    {
      lastRequestTime_ = now;
    }
    //cout << "lastRequestTime = " << lastRequestTime_.tv_sec
    //     << " " << lastRequestTime_.tv_usec << endl;

    ReadData data;
    do {
      CURL* han = curl_easy_init();

      if(han==0)
      {
        cerr << "could not create handle" << endl;
        // this will end cmsRun 
        return std::auto_ptr<edm::EventPrincipal>();
      }

      setopt(han,CURLOPT_URL,eventurl_);
      setopt(han,CURLOPT_WRITEFUNCTION,func);
      setopt(han,CURLOPT_WRITEDATA,&data);

      // 24-Aug-2006, KAB: send our consumer ID as part of the event request
      char msgBuff[100];
      OtherMessageBuilder requestMessage(&msgBuff[0], Header::EVENT_REQUEST,
                                         sizeof(char_uint32));
      uint8 *bodyPtr = requestMessage.msgBody();
      char_uint32 convertedId;
      convert(consumerId_, convertedId);
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
        cerr << "curl perform failed for event" << endl;
        // this will end cmsRun 
        return std::auto_ptr<edm::EventPrincipal>();
      }
      if(data.d_.length() == 0)
      {
        std::cout << "...waiting for event from Storage Manager..." << std::endl;
        // sleep for the standard request interval
        usleep(static_cast<int>(1000000 * minEventRequestInterval_));
      }
    } while (data.d_.length() == 0);

    int len = data.d_.length();
    FDEBUG(9) << "EventStreamHttpReader received len = " << len << std::endl;
    buf_.resize(len);
    for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

    // first check if done message
    // need to use this BUT does EventMsgView EVER look like a MsgCode DONE message!!!
    //edm::MsgCode msgtest(&buf_[0],len);
    //if(msgtest.getCode() == MsgCode::DONE) {
    // OtherMessageView class not working
    OtherMessageView msgView(&buf_[0]);
    //std::cout << "received other message code = " << msgView.code()
    //          << " and size = " << msgView.size()
    //          << " and check against " << Header::DONE << endl;

    if (msgView.code() == Header::DONE) {
      // this will end cmsRun 
      std::cout << "Storage Manager has halted - ending run" << std::endl;
      return std::auto_ptr<edm::EventPrincipal>();
    } else {
      events_read_++;
      //edm::EventMsg msg(&buf_[0],len);
      //return decoder_.decodeEvent(msg,productRegistry());
      EventMsgView eventView(&buf_[0]);
      return deserializeEvent(eventView,productRegistry());
    }
  }

  std::auto_ptr<SendJobHeader> EventStreamHttpReader::readHeader()
  {
    // repeat a http get every 5 seconds until we get the registry
    // do it like this for the proof of principle test
    ReadData data;
    do {
      CURL* han = curl_easy_init();

      if(han==0)
        {
          cerr << "could not create handle" << endl;
          //return 0; //or use this?
          throw cms::Exception("readHeader","EventStreamHttpReader")
            << "Could not get header: probably XDAQ not running on Storage Manager "
            << "\n";
        }

      setopt(han,CURLOPT_URL,headerurl_);
      setopt(han,CURLOPT_WRITEFUNCTION,func);
      setopt(han,CURLOPT_WRITEDATA,&data);

      // 10-Aug-2006, KAB: send our consumer ID as part of the header request
      char msgBuff[100];
      OtherMessageBuilder requestMessage(&msgBuff[0], Header::HEADER_REQUEST,
                                         sizeof(char_uint32));
      uint8 *bodyPtr = requestMessage.msgBody();
      char_uint32 convertedId;
      convert(consumerId_, convertedId);
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
        //return 0; //or use this?
        throw cms::Exception("readHeader","EventStreamHttpReader")
          << "Could not get header: probably XDAQ not running on Storage Manager "
          << "\n";
      }
      if(data.d_.length() == 0)
      {
        std::cout << "...waiting for header from Storage Manager..." << std::endl;
        // sleep for desired amount of time
        sleep(headerRetryInterval_);
      }
    } while (data.d_.length() == 0);

    //JobHeaderDecoder hdecoder;
    std::vector<char> regdata(1000*1000);

    // rely on http transfer string of correct length!
    int len = data.d_.length();
    FDEBUG(9) << "EventStreamHttpReader received registry len = " << len << std::endl;
    regdata.resize(len);
    for (int i=0; i<len ; i++) regdata[i] = data.d_[i];
    //edm::InitMsg msg(&regdata[0],len);
    InitMsgView initView(&regdata[0]);
    //hltBitCount = initView.get_hlt_bit_cnt();
    //l1BitCount = initView.get_l1_bit_cnt();
    // 21-Jun-2006, KAB:  catch (and re-throw) any exceptions decoding
    // the job header so that we can display the returned HTML and
    // (hopefully) give the user a hint as to the cause of the problem.
    std::auto_ptr<SendJobHeader> p;
    try {
      //p = hdecoder.decodeJobHeader(msg);
      p = deserializeRegistry(initView);
    }
    catch (cms::Exception excpt) {
      const unsigned int MAX_DUMP_LENGTH = 1000;
      std::cout << "========================================" << std::endl;
      std::cout << "* Exception decoding the getregdata response from the storage manager!" << std::endl;
      if (data.d_.length() <= MAX_DUMP_LENGTH) {
        std::cout << "* Here is the raw text that was returned:" << std::endl;
        std::cout << data.d_ << std::endl;
      }
      else {
        std::cout << "* Here are the first " << MAX_DUMP_LENGTH <<
          " characters of the raw text that was returned:" << std::endl;
        std::cout << (data.d_.substr(0, MAX_DUMP_LENGTH)) << std::endl;
      }
      std::cout << "========================================" << std::endl;
      throw excpt;
    }
    return p;
  }

  void EventStreamHttpReader::registerWithEventServer()
  {
    ReadData data;
    uint32 registrationStatus;
    do {
      data.d_.clear();
      CURL* han = curl_easy_init();
      if(han==0)
        {
          cerr << "could not create handle" << endl;
          //return 0; //or use this?
          throw cms::Exception("registerWithEventServer","EventStreamHttpReader")
            << "Unable to create curl handle\n";
        }

      // set the standard http request options
      setopt(han,CURLOPT_URL,subscriptionurl_);
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
        //return 0; //or use this?
        throw cms::Exception("registerWithEventServer","EventStreamHttpReader")
          << "Could not register: probably XDAQ not running on Storage Manager "
          << "\n";
      }
      registrationStatus = ConsRegResponseBuilder::ES_NOT_READY;
      if(data.d_.length() > 0)
      {
        int len = data.d_.length();
        FDEBUG(9) << "EventStreamHttpReader received len = " << len << std::endl;
        buf_.resize(len);
        for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

        try {
          ConsRegResponseView respView(&buf_[0]);
          registrationStatus = respView.getStatus();
          consumerId_ = respView.getConsumerId();
        }
        catch (cms::Exception excpt) {
          const unsigned int MAX_DUMP_LENGTH = 1000;
          std::cout << "========================================" << std::endl;
          std::cout << "* Exception decoding the registerWithEventServer response!" << std::endl;
          if (data.d_.length() <= MAX_DUMP_LENGTH) {
            std::cout << "* Here is the raw text that was returned:" << std::endl;
            std::cout << data.d_ << std::endl;
          }
          else {
            std::cout << "* Here are the first " << MAX_DUMP_LENGTH <<
              " characters of the raw text that was returned:" << std::endl;
            std::cout << (data.d_.substr(0, MAX_DUMP_LENGTH)) << std::endl;
          }
          std::cout << "========================================" << std::endl;
          throw excpt;
        }
      }

      if (registrationStatus == ConsRegResponseBuilder::ES_NOT_READY)
      {
        std::cout << "...waiting for registration response from Storage Manager..." << std::endl;
        // sleep for desired amount of time
        sleep(headerRetryInterval_);
      }
    } while (registrationStatus == ConsRegResponseBuilder::ES_NOT_READY);

    FDEBUG(5) << "Consumer ID = " << consumerId_ << endl;
  }
}
