// $Id: StorageManager.cc,v 1.131 2010/08/06 20:24:31 wmtan Exp $
/// @file: StorageManager.cc

#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessor.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentProcessor.h"
#include "EventFilter/StorageManager/interface/SoapUtils.h"
#include "EventFilter/StorageManager/interface/StorageManager.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "i2o/Method.h"
#include "interface/shared/version.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "xcept/tools.h"
#include "xdaq/NamespaceURI.h"
#include "xdata/InfoSpaceFactory.h"
#include "xgi/Method.h"
#include "xoap/Method.h"

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <cstdlib>

using namespace std;
using namespace stor;


StorageManager::StorageManager(xdaq::ApplicationStub * s) :
  xdaq::Application(s),
  _webPageHelper( getApplicationDescriptor(),
    "$Id: StorageManager.cc,v 1.131 2010/08/06 20:24:31 wmtan Exp $ $Name:  $")
{  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making StorageManager");

  // bind all callback functions
  bindI2OCallbacks();
  bindStateMachineCallbacks();
  bindWebInterfaceCallbacks();
  bindConsumerCallbacks();

  std::string errorMsg = "Exception in StorageManager constructor: ";
  try
  {
    // need the line below so that deserializeRegistry can run in
    // order to compare two registries (cannot compare
    // byte-for-byte) (if we keep this) need line below anyway in
    // case we deserialize DQMEvents for collation
    edm::RootAutoLibraryLoader::enable();
    initializeSharedResources();
    _consumerUtils.setSharedResources(_sharedResources);
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    LOG4CPLUS_FATAL( getApplicationLogger(), e.what() );
    XCEPT_RAISE( stor::exception::Exception, e.what() );
  }
  catch(...)
  {
    errorMsg += "unknown exception";
    LOG4CPLUS_FATAL( getApplicationLogger(), errorMsg );
    XCEPT_RAISE( stor::exception::Exception, errorMsg );
  }

  startWorkerThreads();
}


void StorageManager::bindI2OCallbacks()
{
  i2o::bind(this,
            &StorageManager::receiveRegistryMessage,
            I2O_SM_PREAMBLE,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveDataMessage,
            I2O_SM_DATA,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveErrorDataMessage,
            I2O_SM_ERROR,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveDQMMessage,
            I2O_SM_DQM,
            XDAQ_ORGANIZATION_ID);
  i2o::bind(this,
            &StorageManager::receiveEndOfLumiSectionMessage,
            I2O_EVM_LUMISECTION,
            XDAQ_ORGANIZATION_ID);
}


void StorageManager::bindStateMachineCallbacks()
{
  xoap::bind( this,
              &StorageManager::handleFSMSoapMessage,
              "Configure",
              XDAQ_NS_URI );
  xoap::bind( this,
              &StorageManager::handleFSMSoapMessage,
              "Enable",
              XDAQ_NS_URI );
  xoap::bind( this,
              &StorageManager::handleFSMSoapMessage,
              "Stop",
              XDAQ_NS_URI );
  xoap::bind( this,
              &StorageManager::handleFSMSoapMessage,
              "Halt",
              XDAQ_NS_URI );
  xoap::bind( this,
              &StorageManager::handleFSMSoapMessage,
              "EmergencyStop",
              XDAQ_NS_URI );
}


void StorageManager::bindWebInterfaceCallbacks()
{
  xgi::bind(this,&StorageManager::css,                      "styles.css");
  xgi::bind(this,&StorageManager::defaultWebPage,           "Default");
  xgi::bind(this,&StorageManager::storedDataWebPage,        "storedData");
  xgi::bind(this,&StorageManager::rbsenderWebPage,          "rbsenderlist");
  xgi::bind(this,&StorageManager::rbsenderDetailWebPage,    "rbsenderdetail");
  xgi::bind(this,&StorageManager::fileStatisticsWebPage,    "fileStatistics");
  xgi::bind(this,&StorageManager::dqmEventStatisticsWebPage,"dqmEventStatistics");
  xgi::bind(this,&StorageManager::consumerStatisticsPage,   "consumerStatistics" );
  xgi::bind(this,&StorageManager::consumerListWebPage,      "consumerList");
  xgi::bind(this,&StorageManager::throughputWebPage,        "throughputStatistics");
}


void StorageManager::bindConsumerCallbacks()
{
  // event consumers
  xgi::bind( this, &StorageManager::processConsumerRegistrationRequest, "registerConsumer" );
  xgi::bind( this, &StorageManager::processConsumerHeaderRequest, "getregdata" );
  xgi::bind( this, &StorageManager::processConsumerEventRequest, "geteventdata" );

  // dqm event consumers
  xgi::bind(this,&StorageManager::processDQMConsumerRegistrationRequest, "registerDQMConsumer");
  xgi::bind(this,&StorageManager::processDQMConsumerEventRequest, "getDQMeventdata");
}


void StorageManager::initializeSharedResources()
{
  _sharedResources.reset(new SharedResources());

  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  unsigned long instance = getApplicationDescriptor()->getInstance();
  _sharedResources->_configuration.reset(new Configuration(ispace, instance));

  QueueConfigurationParams queueParams =
    _sharedResources->_configuration->getQueueConfigurationParams();
  _sharedResources->_commandQueue.
    reset(new CommandQueue(queueParams._commandQueueSize));
  _sharedResources->_fragmentQueue.
    reset(new FragmentQueue(queueParams._fragmentQueueSize, queueParams._fragmentQueueMemoryLimitMB * 1024*1024));
  _sharedResources->_registrationQueue.
    reset(new RegistrationQueue(queueParams._registrationQueueSize));
  _sharedResources->_streamQueue.
    reset(new StreamQueue(queueParams._streamQueueSize, queueParams._streamQueueMemoryLimitMB * 1024*1024));
  _sharedResources->_dqmEventQueue.
    reset(new DQMEventQueue(queueParams._dqmEventQueueSize, queueParams._dqmEventQueueMemoryLimitMB * 1024*1024));

  _sharedResources->_statisticsReporter.reset(
    new StatisticsReporter(this, _sharedResources)
  );
  _sharedResources->_initMsgCollection.reset(new InitMsgCollection());
  _sharedResources->_diskWriterResources.reset(new DiskWriterResources());
  _sharedResources->_dqmEventProcessorResources.reset(new DQMEventProcessorResources());

  _sharedResources->_statisticsReporter->getThroughputMonitorCollection().setFragmentQueue(_sharedResources->_fragmentQueue);
  _sharedResources->_statisticsReporter->getThroughputMonitorCollection().setStreamQueue(_sharedResources->_streamQueue);
  _sharedResources->_statisticsReporter->getThroughputMonitorCollection().setDQMEventQueue(_sharedResources->_dqmEventQueue);

  _sharedResources->
    _discardManager.reset(new DiscardManager(getApplicationContext(),
                                             getApplicationDescriptor(),
                                             _sharedResources->_statisticsReporter->
                                             getDataSenderMonitorCollection()));

  _sharedResources->_registrationCollection.reset( new RegistrationCollection() );
  EventConsumerMonitorCollection& ecmc = 
    _sharedResources->_statisticsReporter->getEventConsumerMonitorCollection();
  _sharedResources->_eventConsumerQueueCollection.reset( new EventQueueCollection( ecmc ) );

  DQMConsumerMonitorCollection& dcmc = 
    _sharedResources->_statisticsReporter->getDQMConsumerMonitorCollection();
  _sharedResources->_dqmEventConsumerQueueCollection.reset( new DQMEventQueueCollection( dcmc ) );
}


void StorageManager::startWorkerThreads()
{

  // Start the workloops
  try
  {
    _fragmentProcessor = new FragmentProcessor( this, _sharedResources );
    _diskWriter = new DiskWriter(this, _sharedResources);
    _dqmEventProcessor = new DQMEventProcessor(this, _sharedResources);
    _sharedResources->_statisticsReporter->startWorkLoop("theStatisticsReporter");
    _fragmentProcessor->startWorkLoop("theFragmentProcessor");
    _diskWriter->startWorkLoop("theDiskWriter");
    _dqmEventProcessor->startWorkLoop("theDQMEventProcessor");
  }
  catch(xcept::Exception &e)
  {
    _sharedResources->moveToFailedState( e );
  }
  catch(std::exception &e)
  {
    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, e.what());
    _sharedResources->moveToFailedState( sentinelException );
  }
  catch(...)
  {
    std::string errorMsg = "Unknown exception when starting the workloops";
    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState( sentinelException );
  }
}


StorageManager::~StorageManager()
{
  delete _fragmentProcessor;
  delete _diskWriter;
  delete _dqmEventProcessor;
}


/////////////////////////////
// I2O call back functions //
/////////////////////////////

void StorageManager::receiveRegistryMessage(toolbox::mem::Reference *ref)
{
  I2OChain i2oChain(ref);

  // Set the I2O message pool pointer. Only done for init messages.
  ThroughputMonitorCollection& throughputMonCollection =
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection();
  throughputMonCollection.setMemoryPoolPointer( ref->getBuffer()->getPool() );

  FragmentMonitorCollection& fragMonCollection =
    _sharedResources->_statisticsReporter->getFragmentMonitorCollection();
  fragMonCollection.getAllFragmentSizeMQ().addSample( 
    static_cast<double>( i2oChain.totalDataSize() ) / 0x100000
  );

  _sharedResources->_fragmentQueue->enq_wait(i2oChain);
}


void StorageManager::receiveDataMessage(toolbox::mem::Reference *ref)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    _sharedResources->_statisticsReporter->getFragmentMonitorCollection();
  fragMonCollection.addEventFragmentSample( i2oChain.totalDataSize() );

  _sharedResources->_fragmentQueue->enq_wait(i2oChain);

#ifdef STOR_DEBUG_DUPLICATE_MESSAGES
  double r = rand()/static_cast<double>(RAND_MAX);
  if (r < 0.001)
  {
    LOG4CPLUS_INFO(this->getApplicationLogger(), "Simulating duplicated data message");
    receiveDataMessage(ref->duplicate());
  }
#endif
}


void StorageManager::receiveErrorDataMessage(toolbox::mem::Reference *ref)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    _sharedResources->_statisticsReporter->getFragmentMonitorCollection();
  fragMonCollection.addEventFragmentSample( i2oChain.totalDataSize() );

  _sharedResources->_fragmentQueue->enq_wait(i2oChain);
}


void StorageManager::receiveDQMMessage(toolbox::mem::Reference *ref)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    _sharedResources->_statisticsReporter->getFragmentMonitorCollection();
  fragMonCollection.addDQMEventFragmentSample( i2oChain.totalDataSize() );

  _sharedResources->_fragmentQueue->enq_wait(i2oChain);
}


void StorageManager::receiveEndOfLumiSectionMessage(toolbox::mem::Reference *ref)
{
  I2OChain i2oChain( ref );

  FragmentMonitorCollection& fragMonCollection =
    _sharedResources->_statisticsReporter->getFragmentMonitorCollection();
  fragMonCollection.addFragmentSample( i2oChain.totalDataSize() );

  RunMonitorCollection& runMonCollection =
    _sharedResources->_statisticsReporter->getRunMonitorCollection();
  runMonCollection.getEoLSSeenMQ().addSample( i2oChain.lumiSection() );

  _sharedResources->_streamQueue->enq_wait( i2oChain );
}


///////////////////////////////////////
// Web interface call back functions //
///////////////////////////////////////

void StorageManager::css(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  _webPageHelper.css(in,out);
}


void StorageManager::defaultWebPage(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the default webpage";
  
  try
  {
    _webPageHelper.defaultWebPage(
      out,
      _sharedResources
    );
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


void StorageManager::storedDataWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the stored data webpage";

  try
  {
    _webPageHelper.storedDataWebPage(
      out,
      _sharedResources
    );
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


void StorageManager::consumerStatisticsPage( xgi::Input* in,
                                             xgi::Output* out )
  throw( xgi::exception::Exception )
{

  std::string err_msg =
    "Failed to create consumer statistics page";

  try
  {
    _webPageHelper.consumerStatistics( out,
                                       _sharedResources );
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


void StorageManager::rbsenderWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the data sender webpage";

  try
  {
    _webPageHelper.resourceBrokerOverview(out, _sharedResources);
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


void StorageManager::rbsenderDetailWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the data sender webpage";

  try
  {
    long long localRBID = 0;
    cgicc::Cgicc cgiWrapper(in);
    cgicc::const_form_iterator updateRef = cgiWrapper.getElement("id");
    if (updateRef != cgiWrapper.getElements().end())
    {
      std::string idString = updateRef->getValue();
      localRBID = boost::lexical_cast<long long>(idString);
    }

    _webPageHelper.resourceBrokerDetail(out, _sharedResources, localRBID);
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


void StorageManager::fileStatisticsWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the file statistics webpage";

  try
  {
    _webPageHelper.filesWebPage(
      out,
      _sharedResources
    );
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


void StorageManager::dqmEventStatisticsWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the DQM event statistics webpage";

  try
  {
    _webPageHelper.dqmEventWebPage(
      out,
      _sharedResources
    );
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


void StorageManager::throughputWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the throughput statistics webpage";

  try
  {
    _webPageHelper.throughputWebPage(
      out,
      _sharedResources
    );
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


// Leaving in for now but making it return an empty buffer:
void StorageManager::consumerListWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  out->getHTTPResponseHeader().addHeader( "Content-Type",
                                          "application/octet-stream" );
  out->getHTTPResponseHeader().addHeader( "Content-Transfer-Encoding",
                                          "binary" );
  char buff;
  out->write( &buff, 0 );
}


///////////////////////////////////////
// State Machine call back functions //
///////////////////////////////////////

xoap::MessageReference StorageManager::handleFSMSoapMessage( xoap::MessageReference msg )
  throw( xoap::exception::Exception )
{
  std::string errorMsg;
  xoap::MessageReference returnMsg;

  try {
    errorMsg = "Failed to extract FSM event and parameters from SOAP message: ";
    std::string command = soaputils::extractParameters(msg, this);

    errorMsg = "Failed to put a '" + command + "' state machine event into command queue: ";
    if (command == "Configure")
    {
      _sharedResources->_commandQueue->enq_nowait( stor::event_ptr( new stor::Configure() ) );
    }
    else if (command == "Enable")
    {
      if (_sharedResources->_configuration->streamConfigurationHasChanged())
      {
        _sharedResources->_commandQueue->enq_wait( stor::event_ptr( new stor::Reconfigure() ) );
      }
      _sharedResources->_commandQueue->enq_wait( stor::event_ptr( new stor::Enable() ) );
    }
    else if (command == "Stop")
    {
      _sharedResources->_commandQueue->enq_wait( stor::event_ptr( new stor::Stop() ) );
    }
    else if (command == "Halt")
    {
      _sharedResources->_commandQueue->enq_wait( stor::event_ptr( new stor::Halt() ) );
    }
    else if (command == "EmergencyStop")
    {
      _sharedResources->_commandQueue->enq_wait( stor::event_ptr( new stor::EmergencyStop() ) );
    }
    else
    {
      XCEPT_RAISE(stor::exception::StateMachine,
        "Received an unknown state machine event '" + command + "'.");
    }

    errorMsg = "Failed to create FSM SOAP reply message: ";
    returnMsg = soaputils::createFsmSoapResponseMsg(command,
      _sharedResources->_statisticsReporter->
      getStateMachineMonitorCollection().externallyVisibleState());
  }
  catch (cms::Exception& e) {
    errorMsg += e.explainSelf();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (xcept::Exception &e) {
    XCEPT_DECLARE_NESTED(xoap::exception::Exception,
      sentinelException, errorMsg, e);
    _sharedResources->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (std::exception& e) {
    errorMsg += e.what();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (...) {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState( sentinelException );
    throw sentinelException;
  }

  return returnMsg;
}


////////////////////////////
//// Consumer callbacks ////
////////////////////////////

void
StorageManager::processConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  _consumerUtils.processConsumerRegistrationRequest(in,out);
}


void
StorageManager::processConsumerHeaderRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  _consumerUtils.processConsumerHeaderRequest(in,out);
}


void
StorageManager::processConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  _consumerUtils.processConsumerEventRequest(in,out);
}


void
StorageManager::processDQMConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  _consumerUtils.processDQMConsumerRegistrationRequest(in,out);
}


void
StorageManager::processDQMConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  _consumerUtils.processDQMConsumerEventRequest(in,out);
}


//////////////////////////////////////////////////////////////////////////
// *** Provides factory method for the instantiation of SM applications //
//////////////////////////////////////////////////////////////////////////
// This macro is depreciated:
XDAQ_INSTANTIATE(StorageManager)

// One should use the XDAQ_INSTANTIATOR() in the header file
// and this one here. But this breaks the backward compatibility,
// as all xml configuration files would have to be changed to use
// 'stor::StorageManager' instead of 'StorageManager'.
// XDAQ_INSTANTIATOR_IMPL(stor::StorageManager)


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
