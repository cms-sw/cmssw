// $Id: DiskWriter.cc,v 1.5 2009/07/10 14:51:12 dshpakov Exp $
/// @file: DiskWriter.cc

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"
#include "EventFilter/StorageManager/interface/EventStreamHandler.h"


using namespace stor;

DiskWriter::DiskWriter(xdaq::Application *app, SharedResourcesPtr sr) :
_app(app),
_sharedResources(sr),
_lastFileTimeoutCheckTime(utils::getCurrentTime()),
_actionIsActive(true)
{
  WorkerThreadParams workerParams =
    _sharedResources->_configuration->getWorkerThreadParams();
  _timeout = (unsigned int) workerParams._DWdeqWaitTime;
}


DiskWriter::~DiskWriter()
{
  // Stop the activity
  _actionIsActive = false;

  // Cancel the workloop (will wait until the action has finished)
  _writingWL->cancel();
}


void DiskWriter::startWorkLoop(std::string workloopName)
{
  try
  {
    std::string identifier = utils::getIdentifier(_app->getApplicationDescriptor());
    
    _writingWL = toolbox::task::getWorkLoopFactory()->
      getWorkLoop( identifier + workloopName, "waiting" );
    
    if ( ! _writingWL->isActive() )
    {
      toolbox::task::ActionSignature* processAction = 
        toolbox::task::bind(this, &DiskWriter::writeAction, 
          identifier + "WriteNextEvent");
      _writingWL->submit(processAction);
      
      _writingWL->activate();
    }
  }
  catch (xcept::Exception& e)
  {
    std::string msg = "Failed to start workloop 'DiskWriter' with 'writeNextEvent'.";
    XCEPT_RETHROW(stor::exception::DiskWriting, msg, e);
  }
}


bool DiskWriter::writeAction(toolbox::task::WorkLoop*)
{
  std::string errorMsg = "Failed to write an event: ";
  
  try
  {
    writeNextEvent();
  }
  catch(xcept::Exception &e)
  {

    LOG4CPLUS_FATAL( _app->getApplicationLogger(),
                     errorMsg << xcept::stdformat_exception_history(e) );

    XCEPT_DECLARE_NESTED( stor::exception::DiskWriting,
                          sentinelException, errorMsg, e );
    _app->notifyQualified( "fatal", sentinelException );

    _sharedResources->moveToFailedState( errorMsg + xcept::stdformat_exception_history(e) );

  }
  catch(std::exception &e)
  {
    errorMsg += e.what();

    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::DiskWriting,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState( errorMsg );
  }
  catch(...)
  {
    errorMsg += "Unknown exception";

    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::DiskWriting,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState( errorMsg );
  }

  return _actionIsActive;
}


void DiskWriter::writeNextEvent()
{
  I2OChain event;
  boost::shared_ptr<StreamQueue> sq = _sharedResources->_streamQueue;
  utils::time_point_t startTime = utils::getCurrentTime();
  if (sq->deq_timed_wait(event, _timeout))
  {
    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addPoppedEventSample(event.totalDataSize());

    _sharedResources->_diskWriterResources->setBusy(true);
    writeEventToStreams(event);

    if ( timeToCheckForFileTimeOut() ) closeTimedOutFiles();
  }
  else
  {
    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);

    closeTimedOutFiles();
    _sharedResources->_diskWriterResources->setBusy(false);
  }

  EvtStrConfigListPtr evtCfgList;
  ErrStrConfigListPtr errCfgList;
  double newTimeoutValue;
  bool doConfig;
  if (_sharedResources->_diskWriterResources->
    streamChangeRequested(doConfig, evtCfgList, errCfgList, newTimeoutValue))
  {
    destroyStreams();
    if (doConfig)
    {
      configureEventStreams(evtCfgList);
      configureErrorStreams(errCfgList);
      _timeout = (unsigned int) newTimeoutValue;
    }
    _sharedResources->_diskWriterResources->streamChangeDone();
  }
}


void DiskWriter::writeEventToStreams(const I2OChain& event)
{
  std::vector<StreamID> streams = event.getStreamTags();
  for (
    std::vector<StreamID>::iterator it = streams.begin(), itEnd = streams.end();
    it != itEnd;
    ++it
  )
  {
    try
    {
      _streamHandlers.at(*it)->writeEvent(event);

      _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDiskWriteSample(event.totalDataSize());
    }
    catch (std::out_of_range& e)
    {
      std::ostringstream msg;
      msg << "Unable to retrieve stream handler for " << (*it) << " : ";
      msg << e.what();
      XCEPT_RAISE(exception::UnknownStreamId, msg.str());
    }
  }
}


void DiskWriter::closeTimedOutFiles()
{
  utils::time_point_t currentTime = utils::getCurrentTime();
  for (
    StreamHandlers::iterator it = _streamHandlers.begin(), itEnd = _streamHandlers.end();
    it != itEnd;
    ++it
  )
  {
    (*it)->closeTimedOutFiles(currentTime);
  }
  _lastFileTimeoutCheckTime  = currentTime;
}


bool DiskWriter::timeToCheckForFileTimeOut()
{
  const DiskWritingParams dwParams =
    _sharedResources->_configuration->getDiskWritingParams();
  utils::time_point_t now = utils::getCurrentTime();
  return ((now - _lastFileTimeoutCheckTime) > dwParams._fileClosingTestInterval);
}


void DiskWriter::configureEventStreams(EvtStrConfigListPtr cfgList)
{
  for (
    EvtStrConfigList::iterator it = cfgList->begin(),
      itEnd = cfgList->end();
    it != itEnd;
    ++it
  ) 
  {
    makeEventStream(*it);
  }
}


void DiskWriter::configureErrorStreams(ErrStrConfigListPtr cfgList)
{
  for (
    ErrStrConfigList::iterator it = cfgList->begin(),
      itEnd = cfgList->end();
    it != itEnd;
    ++it
  ) 
  {
    makeErrorStream(*it);
  }
}


void DiskWriter::makeEventStream(EventStreamConfigurationInfo& streamCfg)
{
  boost::shared_ptr<EventStreamHandler> newHandler(
    new EventStreamHandler(streamCfg, _sharedResources)
  );
  _streamHandlers.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
  streamCfg.setStreamId(_streamHandlers.size() - 1);
}


void DiskWriter::makeErrorStream(ErrorStreamConfigurationInfo& streamCfg)
{
  boost::shared_ptr<FRDStreamHandler> newHandler(
    new FRDStreamHandler(streamCfg, _sharedResources)
  );
  _streamHandlers.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
  streamCfg.setStreamId(_streamHandlers.size() - 1);
}


void DiskWriter::destroyStreams()
{
  _streamHandlers.clear();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
