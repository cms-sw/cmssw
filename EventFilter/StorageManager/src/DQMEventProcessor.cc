// $Id: DQMEventProcessor.cc,v 1.5 2009/07/10 14:51:12 dshpakov Exp $
/// @file: DQMEventProcessor.cc

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessor.h"

using namespace stor;


DQMEventProcessor::DQMEventProcessor(xdaq::Application *app, SharedResourcesPtr sr) :
_app(app),
_sharedResources(sr),
_actionIsActive(true),
_dqmEventStore( sr->_statisticsReporter->getDQMEventMonitorCollection() )
{
  WorkerThreadParams workerParams =
    _sharedResources->_configuration->getWorkerThreadParams();
  _timeout = static_cast<unsigned int>(workerParams._DQMEPdeqWaitTime);
}


DQMEventProcessor::~DQMEventProcessor()
{
  // Stop the activity
  _actionIsActive = false;

  // Cancel the workloop (will wait until the action has finished)
  _processWL->cancel();
}


void DQMEventProcessor::startWorkLoop(std::string workloopName)
{
  try
  {
    std::string identifier = utils::getIdentifier(_app->getApplicationDescriptor());
    
    _processWL = toolbox::task::getWorkLoopFactory()->
      getWorkLoop( identifier + workloopName, "waiting" );
    
    if ( ! _processWL->isActive() )
    {
      toolbox::task::ActionSignature* processAction = 
        toolbox::task::bind(this, &DQMEventProcessor::processDQMEvents,
          identifier + "ProcessNextDQMEvent");
      _processWL->submit(processAction);
      
      _processWL->activate();
    }
  }
  catch (xcept::Exception& e)
  {
    std::string msg = "Failed to start workloop 'DQMEventProcessor' with 'processNextDQMEvent'.";
    XCEPT_RETHROW(stor::exception::DQMEventProcessing, msg, e);
  }
}


bool DQMEventProcessor::processDQMEvents(toolbox::task::WorkLoop*)
{
  std::string errorMsg = "Failed to process a DQM event: ";
  
  try
  {
    processNextDQMEvent();
  }
  catch(xcept::Exception &e)
  {
    LOG4CPLUS_FATAL( _app->getApplicationLogger(),
                     errorMsg << xcept::stdformat_exception_history(e) );

    XCEPT_DECLARE_NESTED( stor::exception::DQMEventProcessing,
                          sentinelException, errorMsg, e );
    _app->notifyQualified( "fatal", sentinelException );
    
    _sharedResources->moveToFailedState( errorMsg + xcept::stdformat_exception_history(e) );
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    
    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::DQMEventProcessing,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);
    
    _sharedResources->moveToFailedState( errorMsg );
  }
  catch(...)
  {
    errorMsg += "Unknown exception";
    
    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::DQMEventProcessing,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState( errorMsg );
  }

  return _actionIsActive;
}


void DQMEventProcessor::processNextDQMEvent()
{
  I2OChain dqmEvent;
  boost::shared_ptr<DQMEventQueue> eq = _sharedResources->_dqmEventQueue;
  utils::time_point_t startTime = utils::getCurrentTime();
  if (eq->deq_timed_wait(dqmEvent, _timeout))
  {
    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDQMEventProcessorIdleSample(elapsedTime);
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addPoppedDQMEventSample(dqmEvent.totalDataSize());

    _dqmEventStore.addDQMEvent(dqmEvent);
  }
  else
  {
    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDQMEventProcessorIdleSample(elapsedTime);
  }

  processCompleteDQMEventRecords();

  DQMEventProcessorResources::Requests requests;
  DQMProcessingParams dqmParams;
  double newTimeoutValue;
  if (_sharedResources->_dqmEventProcessorResources->
    getRequests(requests, dqmParams, newTimeoutValue))
  {
    if (requests.configuration)
    {
      _timeout = static_cast<unsigned int>(newTimeoutValue);
      _dqmEventStore.setParameters(dqmParams);
      checkDirectories(dqmParams);
    }
    if (requests.endOfRun)
    {
      endOfRun();
    }
    if (requests.storeDestruction)
    {
      _dqmEventStore.clear();
    }
    _sharedResources->_dqmEventProcessorResources->requestsDone();
  }
}

void DQMEventProcessor::endOfRun()
{
  _dqmEventStore.writeAndPurgeAllDQMInstances();
  processCompleteDQMEventRecords();
}


void DQMEventProcessor::processCompleteDQMEventRecords()
{
  DQMEventRecord::GroupRecord dqmRecordEntry;
  while ( _dqmEventStore.getCompletedDQMGroupRecordIfAvailable(dqmRecordEntry) )
  {
    _sharedResources->
      _dqmEventConsumerQueueCollection->addEvent(dqmRecordEntry);
  }
}


void DQMEventProcessor::checkDirectories(DQMProcessingParams const& dqmParams) const
{
  if ( dqmParams._archiveDQM )
  {
    utils::checkDirectory(dqmParams._filePrefixDQM);
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
