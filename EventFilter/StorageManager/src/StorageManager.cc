// $Id: StorageManager.cc,v 1.104 2009/07/10 11:41:04 dshpakov Exp $

#include "EventFilter/StorageManager/interface/ConsumerUtils.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/SoapUtils.h"
#include "EventFilter/StorageManager/interface/StorageManager.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"

#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

#include "i2o/Method.h"
#include "xcept/tools.h"
#include "xdaq/NamespaceURI.h"
#include "xdata/InfoSpaceFactory.h"
#include "xgi/Method.h"
#include "xoap/Method.h"

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace stor;


StorageManager::StorageManager(xdaq::ApplicationStub * s) :
  xdaq::Application(s),
  _webPageHelper( getApplicationDescriptor(),
    "$Id: StorageManager.cc,v 1.104 2009/07/10 11:41:04 dshpakov Exp $ $Name:  $")
{  
  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making StorageManager");

  // bind all callback functions
  bindI2OCallbacks();
  bindStateMachineCallbacks();
  bindWebInterfaceCallbacks();
  bindConsumerCallbacks();

  try
    {
      // need the line below so that deserializeRegistry can run in
      // order to compare two registries (cannot compare
      // byte-for-byte) (if we keep this) need line below anyway in
      // case we deserialize DQMEvents for collation
      edm::RootAutoLibraryLoader::enable();
      initializeSharedResources();
    }
  catch(std::exception &e)
    {
      LOG4CPLUS_FATAL( getApplicationLogger(), e.what() );
      XCEPT_RAISE( stor::exception::Exception, e.what() );
    }
  catch(...)
  {
    std::string errorMsg = "Unknown exception in StorageManager constructor";
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
  xgi::bind(this,&StorageManager::throughputWebPage,"throughputStatistics");
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
    reset(new FragmentQueue(queueParams._fragmentQueueSize));
  _sharedResources->_registrationQueue.
    reset(new RegistrationQueue(queueParams._registrationQueueSize));
  _sharedResources->_streamQueue.
    reset(new StreamQueue(queueParams._streamQueueSize));
  _sharedResources->_dqmEventQueue.
    reset(new DQMEventQueue(queueParams._dqmEventQueueSize));

  _sharedResources->_statisticsReporter.reset(new StatisticsReporter(this));
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
  boost::shared_ptr<ConsumerMonitorCollection>
    cmcptr( _sharedResources->_statisticsReporter->getEventConsumerMonitorCollection() );
  _sharedResources->_eventConsumerQueueCollection.reset( new EventQueueCollection( cmcptr ) );
  cmcptr = _sharedResources->_statisticsReporter->getDQMConsumerMonitorCollection();
  _sharedResources->_dqmEventConsumerQueueCollection.reset( new DQMEventQueueCollection( cmcptr ) );
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
    LOG4CPLUS_FATAL(getApplicationLogger(),
      e.what() << xcept::stdformat_exception_history(e));

    notifyQualified("fatal", e);

    _sharedResources->moveToFailedState( e.what() );
  }
  catch(std::exception &e)
  {
    LOG4CPLUS_FATAL(getApplicationLogger(),
      e.what());
    
    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, e.what());
    notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState( e.what() );
  }
  catch(...)
  {
    std::string errorMsg = "Unknown exception when starting the workloops";
    LOG4CPLUS_FATAL(getApplicationLogger(),
      errorMsg);

    XCEPT_DECLARE(stor::exception::Exception,
      sentinelException, errorMsg);
    notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState( errorMsg );
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
  ResourceMonitorCollection& resourceMonCollection =
    _sharedResources->_statisticsReporter->getResourceMonitorCollection();
  resourceMonCollection.setMemoryPoolPointer( ref->getBuffer()->getPool() );

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
    LOG4CPLUS_FATAL( getApplicationLogger(), errorMsg );

    _sharedResources->moveToFailedState( errorMsg );

    XCEPT_RAISE(xoap::exception::Exception, errorMsg);
  }
  catch (xcept::Exception &e) {
    LOG4CPLUS_FATAL( getApplicationLogger(),
      errorMsg << xcept::stdformat_exception_history(e));

    _sharedResources->moveToFailedState( errorMsg + xcept::stdformat_exception_history(e) );

    XCEPT_RETHROW(xoap::exception::Exception, errorMsg, e);
  }
  catch (std::exception& e) {
    errorMsg += e.what();
    LOG4CPLUS_FATAL( getApplicationLogger(), errorMsg );

    _sharedResources->moveToFailedState( errorMsg );

    XCEPT_RAISE(xoap::exception::Exception, errorMsg);
  }
  catch (...) {
    errorMsg += "Unknown exception";

    LOG4CPLUS_FATAL( getApplicationLogger(), errorMsg );

    _sharedResources->moveToFailedState( errorMsg );

    XCEPT_RAISE(xoap::exception::Exception, errorMsg);
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

  // Get consumer ID if registration is allowed:
  ConsumerID cid = _sharedResources->_registrationCollection->getConsumerID();

  if( !cid.isValid() )
    {
      writeNotReady( out );
      return;
    }

  const utils::duration_t secs2stale =
    _sharedResources->_configuration->getEventServingParams()._activeConsumerTimeout;
  const enquing_policy::PolicyTag policy = enquing_policy::DiscardOld;
  const size_t qsize =
    _sharedResources->_configuration->getEventServingParams()._consumerQueueSize;

  // Create registration info and set consumer ID:
  stor::ConsRegPtr reginfo;
  std::string errorMsg = "Error parsing an event consumer registration request";
  try
    {
      reginfo = parseEventConsumerRegistration( in, qsize, policy, secs2stale );
    }
  catch ( edm::Exception& excpt )
    {
      errorMsg.append( ": " );
      errorMsg.append( excpt.what() );

      LOG4CPLUS_ERROR(this->getApplicationLogger(), errorMsg);

      XCEPT_DECLARE(stor::exception::ConsumerRegistration,
      sentinelException, errorMsg);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  catch ( xcept::Exception& excpt )
    {
      LOG4CPLUS_ERROR(this->getApplicationLogger(),
        errorMsg << xcept::stdformat_exception_history(excpt));

      XCEPT_DECLARE_NESTED(stor::exception::ConsumerRegistration,
      sentinelException, errorMsg, excpt);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  catch ( ... )
    {
      errorMsg.append( ": unknown exception" );

      LOG4CPLUS_ERROR(this->getApplicationLogger(), errorMsg);

      XCEPT_DECLARE(stor::exception::ConsumerRegistration,
      sentinelException, errorMsg);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  reginfo->setConsumerID( cid );

  // Create queue and set queue ID:
  QueueID qid =
    _sharedResources->_eventConsumerQueueCollection->createQueue( cid,
                                                                  policy,
                                                                  qsize,
                                                                  secs2stale );
  if( !qid.isValid() )
    {
      writeNotReady( out );
      return;
    }

  reginfo->setQueueID( qid );

  // Register consumer with InitMsgCollection:
  bool reg_ok =
    _sharedResources->_initMsgCollection->registerConsumer( cid,
                                                            reginfo->selHLTOut() );
  if( !reg_ok )
    {
      writeNotReady( out );
      return;
    }

  // Add registration to collection:
  bool add_ok = 
    _sharedResources->_registrationCollection->addRegistrationInfo( cid,
                                                                    reginfo );
  if( !add_ok )
    {
      writeNotReady( out );
      return;
    }

  // Put registration on the queue:
  _sharedResources->_registrationQueue->enq_wait( reginfo );

  // Reply to consumer:
  writeConsumerRegistration( out, cid );

}


void
StorageManager::processConsumerHeaderRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{

  ConsumerID cid = getConsumerID( in );
  if( !cid.isValid() )
    {
      writeEmptyBuffer( out );
      return;
    }

  // 20-Apr-2009, KAB - treat the proxy server like any other consumer. If
  // and when we need to support multiple HLT output modules with the proxy
  // server, then we can go back to sending the full InitMsgCollection.
  InitMsgSharedPtr payload =
    _sharedResources->_initMsgCollection->getElementForConsumer( cid );

  if( payload.get() == NULL )
    {
      writeEmptyBuffer( out );
      return;
    }

  writeConsumerHeader( out, payload );

}


void
StorageManager::processConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{

  ConsumerID cid = getConsumerID( in );
  if( !cid.isValid() )
    {
      writeEmptyBuffer( out );
      return;
    }

  if ( !_sharedResources->_registrationCollection->registrationIsAllowed() )
    {
      writeDone( out );
      return;
    }

  I2OChain evt =
    _sharedResources->_eventConsumerQueueCollection->popEvent( cid );
  if( evt.faulty() )
    {
      writeEmptyBuffer( out );
      return;
    }

  writeConsumerEvent( out, evt );
}


void
StorageManager::processDQMConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{

  // Get consumer ID if registration is allowed:
  ConsumerID cid = _sharedResources->_registrationCollection->getConsumerID();

  if( !cid.isValid() )
    {
      writeNotReady( out );
      return;
    }

  const utils::duration_t secs2stale =
    _sharedResources->_configuration->getEventServingParams()._DQMactiveConsumerTimeout;
  const enquing_policy::PolicyTag policy = enquing_policy::DiscardOld;
  const size_t qsize =
    _sharedResources->_configuration->getEventServingParams()._consumerQueueSize;

  // Create registration info and set consumer ID:
  stor::DQMEventConsRegPtr dqmreginfo;
  std::string errorMsg = "Error parsing a DQM event consumer registration request";
  try
    {
      dqmreginfo = parseDQMEventConsumerRegistration( in, qsize, policy, secs2stale );
    }
  catch ( edm::Exception& excpt )
    {
      errorMsg.append( ": " );
      errorMsg.append( excpt.what() );

      LOG4CPLUS_ERROR(this->getApplicationLogger(), errorMsg);

      XCEPT_DECLARE(stor::exception::DQMConsumerRegistration,
      sentinelException, errorMsg);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  catch ( xcept::Exception& excpt )
    {
      LOG4CPLUS_ERROR(this->getApplicationLogger(),
        errorMsg << xcept::stdformat_exception_history(excpt));

      XCEPT_DECLARE_NESTED(stor::exception::DQMConsumerRegistration,
      sentinelException, errorMsg, excpt);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  catch ( ... )
    {
      errorMsg.append( ": unknown exception" );

      LOG4CPLUS_ERROR(this->getApplicationLogger(), errorMsg);

      XCEPT_DECLARE(stor::exception::DQMConsumerRegistration,
      sentinelException, errorMsg);
      notifyQualified("error", sentinelException);

      writeErrorString( out, errorMsg );
      return;
    }
  dqmreginfo->setConsumerID( cid );

  // Create queue and set queue ID:
  QueueID qid =
    _sharedResources->_dqmEventConsumerQueueCollection->createQueue( cid,
                                                                     policy,
                                                                     qsize,
                                                                     secs2stale );
  if( !qid.isValid() )
    {
      writeNotReady( out );
      return;
    }

  dqmreginfo->setQueueID( qid );

  // Add registration to collection:
  bool add_ok = 
    _sharedResources->_registrationCollection->addRegistrationInfo( cid,
                                                                    dqmreginfo );
  if( !add_ok )
    {
      writeNotReady( out );
      return;
    }

  // Put registration on the queue:
  _sharedResources->_registrationQueue->enq_wait( dqmreginfo );

  // Reply to consumer:
  writeConsumerRegistration( out, cid );

}


void
StorageManager::processDQMConsumerEventRequest( xgi::Input* in, xgi::Output* out )
  throw( xgi::exception::Exception )
{

  ConsumerID cid = getConsumerID( in );
  if( !cid.isValid() )
    {
      writeEmptyBuffer( out );
      return;
    }

  if ( !_sharedResources->_registrationCollection->registrationIsAllowed() )
    {
      writeDone( out );
      return;
    }

  DQMEventRecord::GroupRecord dqmGroupRecord =
    _sharedResources->_dqmEventConsumerQueueCollection->popEvent( cid );

  if ( !dqmGroupRecord.empty() )
    writeDQMConsumerEvent( out, dqmGroupRecord.getDQMEventMsgView() );
  else
    writeEmptyBuffer( out );
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
