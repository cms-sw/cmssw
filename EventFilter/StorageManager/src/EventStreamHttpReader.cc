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

  $Id: EventStreamHttpReader.cc,v 1.40 2010/05/17 15:59:10 mommsen Exp $
/// @file: EventStreamHttpReader.cc
*/

#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "EventFilter/StorageManager/interface/SMCurlInterface.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/ConsRegMessage.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include <algorithm>
#include <iterator>
#include "curl/curl.h"

#include <wait.h>

using namespace std;
using namespace edm;

namespace edm
{  
  EventStreamHttpReader::EventStreamHttpReader(edm::ParameterSet const& ps,
                                               edm::InputSourceDescription const& desc):
    edm::StreamerInputSource(ps, desc),
    sourceurl_(ps.getParameter<std::string>("sourceURL")),
    buf_(1000*1000*7), 
    endRunAlreadyNotified_(true),
    runEnded_(false),
    alreadySaidHalted_(false),
    maxConnectTries_(DEFAULT_MAX_CONNECT_TRIES),
    connectTrySleepTime_(DEFAULT_CONNECT_TRY_SLEEP_TIME)
  {
    // Retry connection params (wb)
    maxConnectTries_ = ps.getUntrackedParameter<int>("maxConnectTries",
                                               DEFAULT_MAX_CONNECT_TRIES);
    connectTrySleepTime_ = ps.getUntrackedParameter<int>("connectTrySleepTime",
                                               DEFAULT_CONNECT_TRY_SLEEP_TIME);
    inputFileTransitionsEachEvent_ =
      ps.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", true);

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
    consumerName_ = ps.getUntrackedParameter<std::string>("consumerName","Unknown");
    consumerPriority_ = ps.getUntrackedParameter<std::string>("consumerPriority","normal");
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

    // 26-Jan-2009, KAB: an ugly hack to get ParameterSet to serialize
    // the parameters that we need
    ParameterSet psCopy(ps.toString());
    psCopy.addParameter<double>("TrackedMaxRate", maxEventRequestRate);
    std::string hltOMLabel = ps.getUntrackedParameter<std::string>("SelectHLTOutput",
                                                                   std::string());
    psCopy.addParameter<std::string>("TrackedHLTOutMod", hltOMLabel);
    edm::ParameterSet selectEventsParamSet =
      ps.getUntrackedParameter("SelectEvents", edm::ParameterSet());
    if (! selectEventsParamSet.empty()) {
      Strings path_specs = 
        selectEventsParamSet.getParameter<Strings>("SelectEvents");
      if (! path_specs.empty()) {
        psCopy.addParameter<Strings>("TrackedEventSelection", path_specs);
      }
    }
    std::string trigSelector_ = ps.getUntrackedParameter("TriggerSelector",std::string());
    psCopy.addParameter<std::string>("TriggerSelector",trigSelector_);

    // 28-Aug-2006, KAB: save our parameter set in string format to
    // be sent to the event server to specify our "request" (that is, which
    // events we are interested in).
    consumerPSetString_ = psCopy.toString();

    // 16-Aug-2006, KAB: register this consumer with the event server
    consumerId_ = (time(0) & 0xffffff);  // temporary - will get from ES later
    registerWithEventServer();

    readHeader();
  }

  EventStreamHttpReader::~EventStreamHttpReader()
  {
  }

  edm::EventPrincipal* EventStreamHttpReader::read()
  {
    // repeat a http get every N seconds until we get an event
    // wait for Storage Manager event server buffer to not be empty
    // only way to stop is specify a maxEvents parameter
    // or kill the Storage Manager so the http get fails.

    // try to get an event repeat until we get one, this allows
    // re-registration if the SM is halted or stopped

    bool gotEvent = false;
    edm::EventPrincipal* result = 0;
    while ((!gotEvent) && (!runEnded_) && (!edm::shutdown_flag))
    {
       result = getOneEvent();
       if(result != 0) gotEvent = true;
    }
    // need next line so we only return a null pointer once for each end of run
    if(runEnded_) runEnded_ = false;
    return result;
  }

  edm::EventPrincipal* EventStreamHttpReader::getOneEvent()
  {
    // repeat a http get every N seconds until we get an event
    // wait for Storage Manager event server buffer to not be empty
    // only way to stop is specify a maxEvents parameter or cntrol-c.
    // If the Storage Manager is killed so the http get fails, we
    // end the job as we would be in an unknown state (If SM is up
    // and we have a network problem we just try to get another event,
    // but if SM is killed/dead we want to register.)

    // check if we need to sleep (to enforce the allowed request rate)
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeDiff = (double) now.tv_sec;
    timeDiff -= (double) lastRequestTime_.tv_sec;
    timeDiff += ((double) now.tv_usec / 1000000.0);
    timeDiff -= ((double) lastRequestTime_.tv_usec / 1000000.0);
    if (timeDiff < minEventRequestInterval_)
    {
      double sleepTime = minEventRequestInterval_ - timeDiff;
      // trim off a little sleep time to account for the time taken by
      // calling gettimeofday again
      sleepTime -= 0.01;
      if (sleepTime < 0.0) {sleepTime = 0.0;}
      //std::cout << "sleeping for " << sleepTime << std::endl;
      usleep(static_cast<int>(1000000 * sleepTime));
      gettimeofday(&lastRequestTime_, &dummyTZ);
    }
    else
    {
      lastRequestTime_ = now;
    }

    stor::ReadData data;
    bool alreadySaidWaiting = false;
    do {
      CURL* han = curl_easy_init();

      if(han==0)
      {
        std::cerr << "could not create handle" << std::endl;
        // this will end cmsRun 
        //return std::auto_ptr<edm::EventPrincipal>();
        throw cms::Exception("getOneEvent","EventStreamHttpReader")
            << "Could not get event: problem with curl"
            << "\n";
      }

      stor::setopt(han,CURLOPT_URL,eventurl_);
      stor::setopt(han,CURLOPT_WRITEFUNCTION,stor::func);
      stor::setopt(han,CURLOPT_WRITEDATA,&data);

      // 24-Aug-2006, KAB: send our consumer ID as part of the event request
      char msgBuff[100];
      OtherMessageBuilder requestMessage(&msgBuff[0], Header::EVENT_REQUEST,
                                         sizeof(char_uint32));
      uint8 *bodyPtr = requestMessage.msgBody();
      convert(consumerId_, bodyPtr);
      stor::setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
      stor::setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
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
        std::cerr << "curl perform failed for event, messageStatus = "
             << messageStatus << std::endl;
        // this will end cmsRun 
        //return std::auto_ptr<edm::EventPrincipal>();
        throw cms::Exception("getOneEvent","EventStreamHttpReader")
            << "Could not get event: probably XDAQ not running on Storage Manager "
            << "\n";
      }
      if(data.d_.length() == 0)
      {
        if(!alreadySaidWaiting) {
          std::cout << "...waiting for event from Storage Manager..." << std::endl;
          alreadySaidWaiting = true;
        }
        // sleep for the standard request interval
        usleep(static_cast<int>(1000000 * minEventRequestInterval_));
      }
    } while (data.d_.length() == 0 && !edm::shutdown_flag);
    if (edm::shutdown_flag) {
        return 0;
    }

    int len = data.d_.length();
    FDEBUG(9) << "EventStreamHttpReader received len = " << len << std::endl;
    buf_.resize(len);
    for (int i=0; i<len ; i++) buf_[i] = data.d_[i];

    // first check if done message
    OtherMessageView msgView(&buf_[0]);

    if (msgView.code() == Header::DONE) {
      // no need to register again as the SM/EventServer is kept alive on a stopAction
      // *BUT* for a haltAction, we need a code to say when SM is halted as then we need 
      // register again else the consumerId is wrong and we may get wrong events!
      // We may even need to end the job if a new run has new triggers, etc.
      if(!alreadySaidHalted_) {
        alreadySaidHalted_ = true;
        std::cout << "Storage Manager has stopped - waiting for restart" << std::endl;
        std::cout << "Warning! If you are waiting forever at: "
                  << "...waiting for event from Storage Manager... " << std::endl
                  << "   it may be that the Storage Manager has been halted with a haltAction," << std::endl
                  << "   instead of a stopAction. In this case you should control-c to end " << std::endl
                  << "   this consumer and restart it. (This will be fixed in a future update)" << std::endl;
      }
      // decide if we need to notify that a run has ended
      if(!endRunAlreadyNotified_) {
        endRunAlreadyNotified_ = true;
        setEndRun();
        runEnded_ = true;
      }
      return 0;
    } else {
      // reset need-to-set-end-run flag when we get the first event (here any event)
      endRunAlreadyNotified_ = false;
      alreadySaidHalted_ = false;

      // 29-Jan-2008, KAB:  catch (and re-throw) any exceptions decoding
      // the event data so that we can display the returned HTML and
      // (hopefully) give the user a hint as to the cause of the problem.
      edm::EventPrincipal* evtPtr = 0;
      try {
        HeaderView hdrView(&buf_[0]);
        if (hdrView.code() != Header::EVENT) {
          throw cms::Exception("EventStreamHttpReader", "readOneEvent");
        }
        EventMsgView eventView(&buf_[0]);
        evtPtr = deserializeEvent(eventView);
      }
      catch (cms::Exception excpt) {
        const unsigned int MAX_DUMP_LENGTH = 2000;
        std::cout << "========================================" << std::endl;
        std::cout << "* Exception decoding the geteventdata response from the storage manager!" << std::endl;
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
      return evtPtr;
    }
  }

  void EventStreamHttpReader::readHeader()
  {
    // repeat a http get every 5 seconds until we get the registry
    // do it like this for pull mode
    bool alreadySaidWaiting = false;
    stor::ReadData data;
    do {
      CURL* han = curl_easy_init();

      if(han==0)
        {
          std::cerr << "could not create handle" << std::endl;
          //return 0; //or use this?
          throw cms::Exception("readHeader","EventStreamHttpReader")
            << "Could not get header: probably XDAQ not running on Storage Manager "
            << "\n";
        }

      stor::setopt(han,CURLOPT_URL,headerurl_);
      stor::setopt(han,CURLOPT_WRITEFUNCTION,stor::func);
      stor::setopt(han,CURLOPT_WRITEDATA,&data);

      // 10-Aug-2006, KAB: send our consumer ID as part of the header request
      char msgBuff[100];
      OtherMessageBuilder requestMessage(&msgBuff[0], Header::HEADER_REQUEST,
                                         sizeof(char_uint32));
      uint8 *bodyPtr = requestMessage.msgBody();
      convert(consumerId_, bodyPtr);
      stor::setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
      stor::setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
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
        std::cerr << "curl perform failed for header" << std::endl;
        // do not retry curl here as we should return to registration instead if we
        // want an automatic recovery
        throw cms::Exception("readHeader","EventStreamHttpReader")
          << "Could not get header: probably XDAQ not running on Storage Manager "
          << "\n";
      }
      if(data.d_.length() == 0)
      {
        if(!alreadySaidWaiting) {
          std::cout << "...waiting for header from Storage Manager..." << std::endl;
          alreadySaidWaiting = true;
        }
        // sleep for desired amount of time
        sleep(headerRetryInterval_);
      }
    } while (data.d_.length() == 0 && !edm::shutdown_flag);
    if (edm::shutdown_flag) {
      throw cms::Exception("readHeader","EventStreamHttpReader")
          << "The header read was aborted by a shutdown request.\n";
    }

    std::vector<char> regdata(1000*1000);

    // rely on http transfer string of correct length!
    int len = data.d_.length();
    FDEBUG(9) << "EventStreamHttpReader received registry len = " << len << std::endl;
    regdata.resize(len);
    for (int i=0; i<len ; i++) regdata[i] = data.d_[i];
    // 21-Jun-2006, KAB:  catch (and re-throw) any exceptions decoding
    // the job header so that we can display the returned HTML and
    // (hopefully) give the user a hint as to the cause of the problem.
    std::auto_ptr<SendJobHeader> p;
    try {
      HeaderView hdrView(&regdata[0]);
      if (hdrView.code() != Header::INIT) {
        throw cms::Exception("EventStreamHttpReader", "readHeader");
      }
      InitMsgView initView(&regdata[0]);
      deserializeAndMergeWithRegistry(initView);
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
  }

  void EventStreamHttpReader::registerWithEventServer()
  {
    stor::ReadData data;
    uint32_t registrationStatus = ConsRegResponseBuilder::ES_NOT_READY;
    bool alreadySaidWaiting = false;
    do {
      data.d_.clear();
      CURL* han = curl_easy_init();
      if(han==0)
        {
          std::cerr << "could not create handle" << std::endl;
          throw cms::Exception("registerWithEventServer","EventStreamHttpReader")
            << "Unable to create curl handle\n";
        }

      // set the standard http request options
      stor::setopt(han,CURLOPT_URL,subscriptionurl_);
      stor::setopt(han,CURLOPT_WRITEFUNCTION,stor::func);
      stor::setopt(han,CURLOPT_WRITEDATA,&data);

      // build the registration request message to send to the storage manager
      const int BUFFER_SIZE = 2000;
      char msgBuff[BUFFER_SIZE];
      ConsRegRequestBuilder requestMessage(msgBuff, BUFFER_SIZE, consumerName_,
                                       consumerPriority_, consumerPSetString_);

      // add the request message as a http post
      stor::setopt(han, CURLOPT_POSTFIELDS, requestMessage.startAddress());
      stor::setopt(han, CURLOPT_POSTFIELDSIZE, requestMessage.size());
      struct curl_slist *headers=NULL;
      headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
      headers = curl_slist_append(headers, "Content-Transfer-Encoding: binary");
      stor::setopt(han, CURLOPT_HTTPHEADER, headers);

      // send the HTTP POST, read the reply, and cleanup before going on
      //CURLcode messageStatus = (CURLcode)-1;
      // set messageStatus to a non-zero (but still within CURLcode enum list)
      CURLcode messageStatus = CURLE_COULDNT_CONNECT;
      int tries = 0;
      while (messageStatus!=0 && !edm::shutdown_flag)
      {
        tries++;
        messageStatus = curl_easy_perform(han);
        if ( messageStatus != 0 )
        {
          if ( tries >= maxConnectTries_ )
          {
            std::cerr << "Giving up waiting for connection after " << tries 
                      << " tries"  << std::endl;
            curl_slist_free_all(headers);
            curl_easy_cleanup(han);
            std::cerr << "curl perform failed for registration" << std::endl;
            throw cms::Exception("registerWithEventServer","EventStreamHttpReader")
              << "Could not register: probably XDAQ not running on Storage Manager "
              << "\n";
          }
          else
          {
            std::cout << "Waiting for connection to StorageManager... " 
                      << tries << "/" << maxConnectTries_
                      << std::endl;
            sleep(connectTrySleepTime_);
          }
        }
      }
      if (edm::shutdown_flag) {
          continue;
      }

      curl_slist_free_all(headers);
      curl_easy_cleanup(han);

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
        if(!alreadySaidWaiting) {
          std::cout << "...waiting for registration response from Storage Manager..." << std::endl;
          alreadySaidWaiting = true;
        }
        // sleep for desired amount of time
        sleep(headerRetryInterval_);
      }
    } while (registrationStatus == ConsRegResponseBuilder::ES_NOT_READY &&
             !edm::shutdown_flag);
    if (edm::shutdown_flag) {
      throw cms::Exception("registerWithEventServer","EventStreamHttpReader")
          << "Registration was aborted by a shutdown request.\n";
    }

    FDEBUG(5) << "Consumer ID = " << consumerId_ << std::endl;
  }
}
