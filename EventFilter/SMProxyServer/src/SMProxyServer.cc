// $Id: SMProxyServer.cc,v 1.47 2011/08/22 14:21:00 mommsen Exp $
/// @file: SMProxyServer.cc

#include "EventFilter/SMProxyServer/interface/Exception.h"
#include "EventFilter/SMProxyServer/interface/SMProxyServer.h"
#include "EventFilter/SMProxyServer/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/SoapUtils.h"
#include "EventFilter/StorageManager/src/ConsumerUtils.icc"

#include "FWCore/Utilities/interface/EDMException.h"

#include "xcept/tools.h"
#include "xdaq/NamespaceURI.h"
#include "xdata/InfoSpaceFactory.h"
#include "xgi/Method.h"
#include "xoap/Method.h"

#include <cstdlib>

using namespace std;
using namespace smproxy;


SMProxyServer::SMProxyServer(xdaq::ApplicationStub * s) :
  xdaq::Application(s)
{  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making SMProxyServer");

  // bind all callback functions
  bindStateMachineCallbacks();
  bindWebInterfaceCallbacks();
  bindConsumerCallbacks();

  std::string errorMsg = "Exception in SMProxyServer constructor: ";
  try
  {
    initializeSharedResources();
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    LOG4CPLUS_FATAL( getApplicationLogger(), e.what() );
    XCEPT_RAISE( exception::Exception, e.what() );
  }
  catch(...)
  {
    errorMsg += "unknown exception";
    LOG4CPLUS_FATAL( getApplicationLogger(), errorMsg );
    XCEPT_RAISE( exception::Exception, errorMsg );
  }

  startWorkerThreads();
}


void SMProxyServer::bindStateMachineCallbacks()
{
  xoap::bind( this,
              &SMProxyServer::handleFSMSoapMessage,
              "Configure",
              XDAQ_NS_URI );
  xoap::bind( this,
              &SMProxyServer::handleFSMSoapMessage,
              "Enable",
              XDAQ_NS_URI );
  xoap::bind( this,
              &SMProxyServer::handleFSMSoapMessage,
              "Stop",
              XDAQ_NS_URI );
  xoap::bind( this,
              &SMProxyServer::handleFSMSoapMessage,
              "Halt",
              XDAQ_NS_URI );
}


void SMProxyServer::bindWebInterfaceCallbacks()
{
  xgi::bind(this,&SMProxyServer::css,                      "styles.css");
  xgi::bind(this,&SMProxyServer::defaultWebPage,           "Default");
  xgi::bind(this,&SMProxyServer::dataRetrieverWebPage,     "dataRetriever");
  xgi::bind(this,&SMProxyServer::dqmEventStatisticsWebPage,"dqmEventStatistics");
  xgi::bind(this,&SMProxyServer::consumerStatisticsWebPage,"consumerStatistics" );
}


void SMProxyServer::bindConsumerCallbacks()
{
  // event consumers
  xgi::bind( this, &SMProxyServer::processConsumerRegistrationRequest, "registerConsumer" );
  xgi::bind( this, &SMProxyServer::processConsumerHeaderRequest, "getregdata" );
  xgi::bind( this, &SMProxyServer::processConsumerEventRequest, "geteventdata" );

  // dqm event consumers
  xgi::bind(this,&SMProxyServer::processDQMConsumerRegistrationRequest, "registerDQMConsumer");
  xgi::bind(this,&SMProxyServer::processDQMConsumerEventRequest, "getDQMeventdata");
}


void SMProxyServer::initializeSharedResources()
{
  stateMachine_.reset( new StateMachine(this) );
  
  consumerUtils_.reset( new ConsumerUtils_t (
      stateMachine_->getConfiguration(),
      stateMachine_->getRegistrationCollection(),
      stateMachine_->getRegistrationQueue(),
      stateMachine_->getInitMsgCollection(),
      stateMachine_->getEventQueueCollection(),
      stateMachine_->getDQMEventQueueCollection(),
      stateMachine_->getStatisticsReporter()->alarmHandler()
    ) );

  smpsWebPageHelper_.reset( new SMPSWebPageHelper(
      getApplicationDescriptor(), stateMachine_));
}


void SMProxyServer::startWorkerThreads()
{
  // Start the workloops
  try
  {
    stateMachine_->getStatisticsReporter()->startWorkLoop("theStatisticsReporter");
  }
  catch(xcept::Exception &e)
  {
    stateMachine_->moveToFailedState(e);
  }
  catch(std::exception &e)
  {
    XCEPT_DECLARE(exception::Exception,
      sentinelException, e.what());
    stateMachine_->moveToFailedState(sentinelException);
  }
  catch(...)
  {
    std::string errorMsg = "Unknown exception when starting the workloops";
    XCEPT_DECLARE(exception::Exception,
      sentinelException, errorMsg);
    stateMachine_->moveToFailedState(sentinelException);
  }
}


///////////////////////////////////////
// Web interface call back functions //
///////////////////////////////////////

void SMProxyServer::css(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  smpsWebPageHelper_->css(in,out);
}


void SMProxyServer::defaultWebPage(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the default webpage";
  
  try
  {
    smpsWebPageHelper_->defaultWebPage(out);
  }
  catch(std::exception &e)
  {
    errorMsg += ": ";
    errorMsg += e.what();
    
    LOG4CPLUS_ERROR(getApplicationLogger(), errorMsg);
    XCEPT_RAISE(xgi::exception::Exception, errorMsg);
  }
  catch(...)
  {
    errorMsg += ": Unknown exception";
    
    LOG4CPLUS_ERROR(getApplicationLogger(), errorMsg);
    XCEPT_RAISE(xgi::exception::Exception, errorMsg);
  }
}


void SMProxyServer::dataRetrieverWebPage(xgi::Input* in, xgi::Output* out)
throw( xgi::exception::Exception )
{

  std::string err_msg =
    "Failed to create data retriever web page";

  try
  {
    smpsWebPageHelper_->dataRetrieverWebPage(out);
  }
  catch( std::exception &e )
  {
    err_msg += ": ";
    err_msg += e.what();
    LOG4CPLUS_ERROR( getApplicationLogger(), err_msg );
    XCEPT_RAISE( xgi::exception::Exception, err_msg );
  }
  catch(...)
  {
    err_msg += ": Unknown exception";
    LOG4CPLUS_ERROR( getApplicationLogger(), err_msg );
    XCEPT_RAISE( xgi::exception::Exception, err_msg );
  }
}


void SMProxyServer::consumerStatisticsWebPage(xgi::Input* in, xgi::Output* out)
throw( xgi::exception::Exception )
{

  std::string err_msg =
    "Failed to create consumer web page";

  try
  {
    smpsWebPageHelper_->consumerStatisticsWebPage(out);
  }
  catch( std::exception &e )
  {
    err_msg += ": ";
    err_msg += e.what();
    LOG4CPLUS_ERROR( getApplicationLogger(), err_msg );
    XCEPT_RAISE( xgi::exception::Exception, err_msg );
  }
  catch(...)
  {
    err_msg += ": Unknown exception";
    LOG4CPLUS_ERROR( getApplicationLogger(), err_msg );
    XCEPT_RAISE( xgi::exception::Exception, err_msg );
  }
}


void SMProxyServer::dqmEventStatisticsWebPage(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the DQM event statistics webpage";

  try
  {
    smpsWebPageHelper_->dqmEventStatisticsWebPage(out);
  }
  catch(std::exception &e)
  {
    errorMsg += ": ";
    errorMsg += e.what();
    
    LOG4CPLUS_ERROR(getApplicationLogger(), errorMsg);
    XCEPT_RAISE(xgi::exception::Exception, errorMsg);
  }
  catch(...)
  {
    errorMsg += ": Unknown exception";
    
    LOG4CPLUS_ERROR(getApplicationLogger(), errorMsg);
    XCEPT_RAISE(xgi::exception::Exception, errorMsg);
  }
}


///////////////////////////////////////
// State Machine call back functions //
///////////////////////////////////////

xoap::MessageReference SMProxyServer::handleFSMSoapMessage( xoap::MessageReference msg )
  throw( xoap::exception::Exception )
{
  std::string errorMsg;
  xoap::MessageReference returnMsg;

  try {
    errorMsg = "Failed to extract FSM event and parameters from SOAP message: ";
    std::string command = stor::soaputils::extractParameters(msg, this);
    std::string newState = "unknown";  

    errorMsg = "Failed to process '" + command + "' state machine event: ";
    if (command == "Configure")
    {
      newState = stateMachine_->processEvent( Configure() );
    }
    else if (command == "Enable")
    {
      newState = stateMachine_->processEvent( Enable() );
    }
    else if (command == "Stop")
    {
      newState = stateMachine_->processEvent( Stop() );
    }
    else if (command == "Halt")
    {
      newState = stateMachine_->processEvent( Halt() );
    }
    else
    {
      XCEPT_RAISE(exception::StateMachine,
        "Received an unknown state machine event '" + command + "'.");
    }

    errorMsg = "Failed to create FSM SOAP reply message: ";
    returnMsg = stor::soaputils::createFsmSoapResponseMsg(command, newState);
  }
  catch (cms::Exception& e) {
    errorMsg += e.explainSelf();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    stateMachine_->moveToFailedState(sentinelException);
  }
  catch (xcept::Exception &e) {
    XCEPT_DECLARE_NESTED(xoap::exception::Exception,
      sentinelException, errorMsg, e);
    stateMachine_->moveToFailedState(sentinelException);
  }
  catch (std::exception& e) {
    errorMsg += e.what();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    stateMachine_->moveToFailedState(sentinelException);
  }
  catch (...) {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    stateMachine_->moveToFailedState(sentinelException);
  }

  return returnMsg;
}


////////////////////////////
//// Consumer callbacks ////
////////////////////////////

void
SMProxyServer::processConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processConsumerRegistrationRequest(in,out);
}


void
SMProxyServer::processConsumerHeaderRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processConsumerHeaderRequest(in,out);
}


void
SMProxyServer::processConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processConsumerEventRequest(in,out);
}


void
SMProxyServer::processDQMConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processDQMConsumerRegistrationRequest(in,out);
}


void
SMProxyServer::processDQMConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processDQMConsumerEventRequest(in,out);
}

namespace stor {
  //////////////////////////////////////
  // Specialization for ConsumerUtils //
  //////////////////////////////////////
  template<>
  void
  ConsumerUtils<smproxy::Configuration,smproxy::EventQueueCollection>::
  writeConsumerEvent(xgi::Output* out, const smproxy::EventMsg& evt) const
  {
    writeHTTPHeaders( out );
    out->write( (char*)evt.dataLocation(), evt.totalDataSize() );
  }
}


//////////////////////////////////////////////////////////////////////////
// *** Provides factory method for the instantiation of SM applications //
//////////////////////////////////////////////////////////////////////////
// This macro is depreciated:
XDAQ_INSTANTIATE(SMProxyServer)

// One should use the XDAQ_INSTANTIATOR() in the header file
// and this one here. But this breaks the backward compatibility,
// as all xml configuration files would have to be changed to use
// 'stor::SMProxyServer' instead of 'SMProxyServer'.
// XDAQ_INSTANTIATOR_IMPL(stor::SMProxyServer)


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
