// $Id: EventServerProxy.h,v 1.8 2012/04/23 08:41:27 mommsen Exp $
/// @file: EventServerProxy.h

#ifndef EventFilter_StorageManager_EventServerProxy_h
#define EventFilter_StorageManager_EventServerProxy_h

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/OtherMessage.h"

#include <string>


namespace stor {

  /**
   * Retrieve events from the Storage Manager event server.
   *
   * This does uses a HTTP get using the CURL library. The Storage Manager
   * event server responses with a binary octet-stream. The init message
   * is also obtained through a HTTP get.
   *
   * $Author: mommsen $
   * $Revision: 1.8 $
   * $Date: 2012/04/23 08:41:27 $
   */

  template<typename RegInfo>
  class EventServerProxy
  {

  public:

    EventServerProxy(edm::ParameterSet const&);
    virtual ~EventServerProxy() {};

    /**
     * Reconnect to the event server
    */
    void reconnect();

    /**
     * Get one event from the event server.
     */
    void getOneEvent(CurlInterface::Content& data);

    /**
     * Try to get one event from the event server.
     * If succesful, returns true.
     */
    bool getEventMaybe(CurlInterface::Content& data);

    /**
     * Get the init message from the the event server.
     */
    void getInitMsg(CurlInterface::Content& data);
    
    
  private:

    void getOneEventFromEventServer(CurlInterface::Content&);
    void checkEvent(CurlInterface::Content&);
    void getInitMsgFromEventServer(CurlInterface::Content&);
    void checkInitMsg(CurlInterface::Content&);
    void registerWithEventServer();
    void connectToEventServer(CurlInterface::Content&);
    bool extractConsumerId(CurlInterface::Content&);

    const RegInfo regInfo_;
    unsigned int consumerId_;
    stor::utils::TimePoint_t nextRequestTime_;
    const stor::utils::Duration_t minEventRequestInterval_;
    
    bool alreadySaidHalted_;
    bool alreadySaidWaiting_;
    unsigned int failedAttemptsToGetData_;
    
  };


  ///////////////////////////////////////////////////////
  // Specializations for EventConsumerRegistrationInfo //
  ///////////////////////////////////////////////////////

  template<>
  inline void
  EventServerProxy<EventConsumerRegistrationInfo>::
  getInitMsgFromEventServer(CurlInterface::Content& data)
  {
    // build the header request message to send to the event server
    char msgBuff[100];
    OtherMessageBuilder requestMessage(
      &msgBuff[0],
      Header::HEADER_REQUEST,
      sizeof(char_uint32)
    );
    uint8 *bodyPtr = requestMessage.msgBody();
    convert(consumerId_, bodyPtr);
    
    // send the header request
    stor::CurlInterfacePtr curl = stor::CurlInterface::getInterface();
    CURLcode result = curl->postBinaryMessage(
      regInfo_.sourceURL() + "/getregdata",
      requestMessage.startAddress(),
      requestMessage.size(),
      data
    );
    
    if ( result != CURLE_OK )
    {
      // connection failed: try to reconnect
      edm::LogError("EventServerProxy") << "curl perform failed for header:"
        << std::string(&data[0]) << std::endl
        << ". Trying to reconnect.";
      data.clear();
      registerWithEventServer();
    }
    
    if( data.empty() )
    {
      if(!alreadySaidWaiting_) {
        edm::LogInfo("EventServerProxy") << "...waiting for header from event server...";
        alreadySaidWaiting_ = true;
      }
      // sleep for desired amount of time
      ::sleep(regInfo_.headerRetryInterval());
    }
    else
    {
      alreadySaidWaiting_ = false;
    }
  }
  
  template<>
  inline void
  EventServerProxy<EventConsumerRegistrationInfo>::
  checkInitMsg(CurlInterface::Content& data)
  {
    try {
      HeaderView hdrView(&data[0]);
      if (hdrView.code() != Header::INIT) {
        throw cms::Exception("EventServerProxy", "readHeader");
      }
    }
    catch (cms::Exception excpt) {
      const unsigned int MAX_DUMP_LENGTH = 1000;
      std::ostringstream dump;
      dump << "========================================" << std::endl;
      dump << "* Exception decoding the getregdata response from the event server!" << std::endl;
      if (data.size() <= MAX_DUMP_LENGTH)
      {
        dump << "* Here is the raw text that was returned:" << std::endl;
        dump << std::string(&data[0]) << std::endl;
      }
      else
      {
        dump << "* Here are the first " << MAX_DUMP_LENGTH <<
          " characters of the raw text that was returned:" << std::endl;
        dump << std::string(&data[0], MAX_DUMP_LENGTH) << std::endl;
      }
      dump << "========================================" << std::endl;
      edm::LogError("EventServerProxy") << dump.str();
      throw excpt;
    }
  }

  template<>
  inline void
  EventServerProxy<EventConsumerRegistrationInfo>::
  getInitMsg(CurlInterface::Content& data)
  {
    do
    {
      data.clear();
      getInitMsgFromEventServer(data);
    }
    while ( !edm::shutdown_flag && data.empty() );
    
    if (edm::shutdown_flag) {
      throw cms::Exception("readHeader","EventServerProxy")
        << "The header read was aborted by a shutdown request.\n";
    }
    
    checkInitMsg(data);
  }

} // namespace stor

#endif // EventFilter_StorageManager_EventServerProxy_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
