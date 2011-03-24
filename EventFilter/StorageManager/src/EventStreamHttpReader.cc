// $Id: EventStreamHttpReader.cc,v 1.44 2011/03/07 15:31:32 mommsen Exp $
/// @file: EventStreamHttpReader.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/OtherMessage.h"

#include <string>

    
namespace stor
{
  ///////////////////////////////////////////////////
  // Specializations for EventServerProxy template //
  ///////////////////////////////////////////////////

  template<>
  void
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
    stor::CurlInterface curl;
    CURLcode result = curl.postBinaryMessage(
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
      sleep(regInfo_.headerRetryInterval());
    }
    else
    {
      alreadySaidWaiting_ = false;
    }
  }
  
  template<>
  void
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
  void
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


namespace edm
{  
  EventStreamHttpReader::EventStreamHttpReader
  (
    ParameterSet const& pset,
    InputSourceDescription const& desc
  ):
  StreamerInputSource(pset, desc),
  eventServerProxy_(pset),
  dropOldLumisectionEvents_(pset.getUntrackedParameter<bool>("dropOldLumisectionEvents", false)),
  lastLS_(0)
  {
    // Default in StreamerInputSource is 'false'
    inputFileTransitionsEachEvent_ =
      pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", true);

    readHeader();
  }
  
  
  EventPrincipal* EventStreamHttpReader::read()
  {
    stor::CurlInterface::Content data;
    unsigned int currentLS(0);
    
    do
    {
      eventServerProxy_.getOneEvent(data);
      if ( data.empty() ) return 0;
      
      HeaderView hdrView(&data[0]);
      if (hdrView.code() == Header::DONE)
      {
        setEndRun();
        return 0;
      }
      
      EventMsgView eventView(&data[0]);
      currentLS = eventView.lumi();
    }
    while (
      dropOldLumisectionEvents_ &&
      lastLS_ > currentLS
    );
    
    lastLS_ = currentLS;
    return deserializeEvent(EventMsgView(&data[0]));
  }
  
  
  void EventStreamHttpReader::readHeader()
  {
    stor::CurlInterface::Content data;
    
    eventServerProxy_.getInitMsg(data);
    InitMsgView initView(&data[0]);
    deserializeAndMergeWithRegistry(initView);
  }

} //namespace edm


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
