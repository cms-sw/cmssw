// $Id: StorageManager.cc,v 1.139 2013/01/07 11:30:00 eulisse Exp $
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
#include "EventFilter/StorageManager/src/ConsumerUtils.icc"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "FWCore/Utilities/interface/EDMException.h"

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
  xdaq::Application(s)
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
    initializeSharedResources();
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
  xgi::bind(this,&StorageManager::inputWebPage,             "input");
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
  sharedResources_.reset(new SharedResources());

  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  unsigned long instance = getApplicationDescriptor()->getInstance();
  sharedResources_->configuration_.reset(new Configuration(ispace, instance));

  QueueConfigurationParams queueParams =
    sharedResources_->configuration_->getQueueConfigurationParams();
  sharedResources_->commandQueue_.
    reset(new CommandQueue(queueParams.commandQueueSize_));
  sharedResources_->fragmentQueue_.
    reset(new FragmentQueue(queueParams.fragmentQueueSize_, queueParams.fragmentQueueMemoryLimitMB_ * 1024*1024));
  sharedResources_->registrationQueue_.
    reset(new RegistrationQueue(queueParams.registrationQueueSize_));
  sharedResources_->streamQueue_.
    reset(new StreamQueue(queueParams.streamQueueSize_, queueParams.streamQueueMemoryLimitMB_ * 1024*1024));
  sharedResources_->dqmEventQueue_.
    reset(new DQMEventQueue(queueParams.dqmEventQueueSize_, queueParams.dqmEventQueueMemoryLimitMB_ * 1024*1024));

  sharedResources_->alarmHandler_.reset( new AlarmHandler(this, sharedResources_) );
  sharedResources_->statisticsReporter_.reset(
    new StatisticsReporter(this, sharedResources_)
  );
  sharedResources_->initMsgCollection_.reset(new InitMsgCollection());
  sharedResources_->diskWriterResources_.reset(new DiskWriterResources());
  sharedResources_->dqmEventProcessorResources_.reset(new DQMEventProcessorResources());

  sharedResources_->statisticsReporter_->getThroughputMonitorCollection().setFragmentQueue(sharedResources_->fragmentQueue_);
  sharedResources_->statisticsReporter_->getThroughputMonitorCollection().setStreamQueue(sharedResources_->streamQueue_);
  sharedResources_->statisticsReporter_->getThroughputMonitorCollection().setDQMEventQueue(sharedResources_->dqmEventQueue_);

  sharedResources_->
    discardManager_.reset(new DiscardManager(getApplicationContext(),
                                             getApplicationDescriptor(),
                                             sharedResources_->statisticsReporter_->
                                             getDataSenderMonitorCollection()));

  sharedResources_->registrationCollection_.reset( new RegistrationCollection() );
  EventConsumerMonitorCollection& ecmc = 
    sharedResources_->statisticsReporter_->getEventConsumerMonitorCollection();
  sharedResources_->eventQueueCollection_.reset( new EventQueueCollection( ecmc ) );

  DQMConsumerMonitorCollection& dcmc = 
    sharedResources_->statisticsReporter_->getDQMConsumerMonitorCollection();
  sharedResources_->dqmEventQueueCollection_.reset( new DQMEventQueueCollection( dcmc ) );

  consumerUtils_.reset( new ConsumerUtils_t(
      sharedResources_->configuration_,
      sharedResources_->registrationCollection_,
      sharedResources_->registrationQueue_,
      sharedResources_->initMsgCollection_,
      sharedResources_->eventQueueCollection_,
      sharedResources_->dqmEventQueueCollection_,
      sharedResources_->alarmHandler_
    ) );

  smWebPageHelper_.reset( new SMWebPageHelper(
      getApplicationDescriptor(), sharedResources_));

}


void StorageManager::startWorkerThreads()
{

  // Start the workloops
  try
  {
    fragmentProcessor_.reset( new FragmentProcessor( this, sharedResources_ ) );
    diskWriter_.reset( new DiskWriter(this, sharedResources_) );
    dqmEventProcessor_.reset( new DQMEventProcessor(this, sharedResources_) );
    sharedResources_->statisticsReporter_->startWorkLoop("theStatisticsReporter");
    fragmentProcessor_->startWorkLoop("theFragmentProcessor");
    diskWriter_->startWorkLoop("theDiskWriter");
    dqmEventProcessor_->startWorkLoop("theDQMEventProcessor");
  }
  catch(xcept::Exception &e)
  {
    sharedResources_->alarmHandler_->moveToFailedState( e );
  }
  catch(std::exception &e)
  {
    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, e.what());
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
  }
  catch(...)
  {
    std::string errorMsg = "Unknown exception when starting the workloops";
    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
  }
}


/////////////////////////////
// I2O call back functions //
/////////////////////////////

void StorageManager::receiveRegistryMessage(toolbox::mem::Reference *ref) throw (i2o::exception::Exception)
{
  I2OChain i2oChain(ref);

  // Set the I2O message pool pointer. Only done for init messages.
  ThroughputMonitorCollection& throughputMonCollection =
    sharedResources_->statisticsReporter_->getThroughputMonitorCollection();
  throughputMonCollection.setMemoryPoolPointer( ref->getBuffer()->getPool() );

  FragmentMonitorCollection& fragMonCollection =
    sharedResources_->statisticsReporter_->getFragmentMonitorCollection();
  fragMonCollection.getAllFragmentSizeMQ().addSample( 
    static_cast<double>( i2oChain.totalDataSize() ) / 0x100000
  );

  sharedResources_->fragmentQueue_->enqWait(i2oChain);
}


void StorageManager::receiveDataMessage(toolbox::mem::Reference *ref) throw (i2o::exception::Exception)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    sharedResources_->statisticsReporter_->getFragmentMonitorCollection();
  fragMonCollection.addEventFragmentSample( i2oChain.totalDataSize() );

  sharedResources_->fragmentQueue_->enqWait(i2oChain);

#ifdef STOR_DEBUG_DUPLICATE_MESSAGES
  double r = rand()/static_cast<double>(RAND_MAX);
  if (r < 0.001)
  {
    LOG4CPLUS_INFO(this->getApplicationLogger(), "Simulating duplicated data message");
    receiveDataMessage(ref->duplicate());
  }
#endif
}


void StorageManager::receiveErrorDataMessage(toolbox::mem::Reference *ref) throw (i2o::exception::Exception)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    sharedResources_->statisticsReporter_->getFragmentMonitorCollection();
  fragMonCollection.addEventFragmentSample( i2oChain.totalDataSize() );

  sharedResources_->fragmentQueue_->enqWait(i2oChain);
}


void StorageManager::receiveDQMMessage(toolbox::mem::Reference *ref) throw (i2o::exception::Exception)
{
  I2OChain i2oChain(ref);

  FragmentMonitorCollection& fragMonCollection =
    sharedResources_->statisticsReporter_->getFragmentMonitorCollection();
  fragMonCollection.addDQMEventFragmentSample( i2oChain.totalDataSize() );

  sharedResources_->fragmentQueue_->enqWait(i2oChain);
}


void StorageManager::receiveEndOfLumiSectionMessage(toolbox::mem::Reference *ref) throw (i2o::exception::Exception)
{
  I2OChain i2oChain( ref );

  FragmentMonitorCollection& fragMonCollection =
    sharedResources_->statisticsReporter_->getFragmentMonitorCollection();
  fragMonCollection.addFragmentSample( i2oChain.totalDataSize() );

  RunMonitorCollection& runMonCollection =
    sharedResources_->statisticsReporter_->getRunMonitorCollection();
  runMonCollection.getEoLSSeenMQ().addSample( i2oChain.lumiSection() );

  sharedResources_->streamQueue_->enqWait( i2oChain );
}


///////////////////////////////////////
// Web interface call back functions //
///////////////////////////////////////

void StorageManager::css(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  smWebPageHelper_->css(in,out);
}


void StorageManager::defaultWebPage(xgi::Input *in, xgi::Output *out)
throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the default webpage";
  
  try
  {
    smWebPageHelper_->defaultWebPage(out);
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


void StorageManager::inputWebPage(xgi::Input *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  std::string errorMsg = "Failed to create the I2O input webpage";

  try
  {
    smWebPageHelper_->inputWebPage(out);
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
    smWebPageHelper_->storedDataWebPage(out);
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
    smWebPageHelper_->consumerStatistics(out);
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
    smWebPageHelper_->resourceBrokerOverview(out);
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

    smWebPageHelper_->resourceBrokerDetail(out, localRBID);
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
    smWebPageHelper_->filesWebPage(out);
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
    smWebPageHelper_->dqmEventWebPage(out);
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
    smWebPageHelper_->throughputWebPage(out);
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
      sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::Configure() ) );
    }
    else if (command == "Enable")
    {
      if (sharedResources_->configuration_->streamConfigurationHasChanged())
      {
        sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::Reconfigure() ) );
      }
      sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::Enable() ) );
    }
    else if (command == "Stop")
    {
      sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::Stop() ) );
    }
    else if (command == "Halt")
    {
      sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::Halt() ) );
    }
    else if (command == "EmergencyStop")
    {
      sharedResources_->commandQueue_->enqWait( stor::EventPtr_t( new stor::EmergencyStop() ) );
    }
    else
    {
      XCEPT_RAISE(stor::exception::StateMachine,
        "Received an unknown state machine event '" + command + "'.");
    }

    errorMsg = "Failed to create FSM SOAP reply message: ";
    returnMsg = soaputils::createFsmSoapResponseMsg(command,
      sharedResources_->statisticsReporter_->
      getStateMachineMonitorCollection().externallyVisibleState());
  }
  catch (cms::Exception& e) {
    errorMsg += e.explainSelf();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (xcept::Exception &e) {
    XCEPT_DECLARE_NESTED(xoap::exception::Exception,
      sentinelException, errorMsg, e);
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (std::exception& e) {
    errorMsg += e.what();
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
    throw sentinelException;
  }
  catch (...) {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE(xoap::exception::Exception,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState( sentinelException );
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
  consumerUtils_->processConsumerRegistrationRequest(in,out);
}


void
StorageManager::processConsumerHeaderRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processConsumerHeaderRequest(in,out);
}


void
StorageManager::processConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processConsumerEventRequest(in,out);
}


void
StorageManager::processDQMConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{
  consumerUtils_->processDQMConsumerRegistrationRequest(in,out);
}


void
StorageManager::processDQMConsumerEventRequest( xgi::Input* in, xgi::Output* out )
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
  ConsumerUtils<Configuration,EventQueueCollection>::
  writeConsumerEvent(xgi::Output* out, const I2OChain& evt) const
  {
    writeHTTPHeaders( out );
    
    #ifdef STOR_DEBUG_CORRUPTED_EVENT_HEADER
    double r = rand()/static_cast<double>(RAND_MAX);
    if (r < 0.1)
    {
      std::cout << "Simulating corrupted event header" << std::endl;
      EventHeader* h = (EventHeader*)evt.dataLocation(0);
      h->protocolVersion_ = 1;
    }
    #endif // STOR_DEBUG_CORRUPTED_EVENT_HEADER
    
    const unsigned int nfrags = evt.fragmentCount();
    for ( unsigned int i = 0; i < nfrags; ++i )
    {
      const unsigned long len = evt.dataSize( i );
      unsigned char* location = evt.dataLocation( i );
      out->write( (char*)location, len );
    } 
  }
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
