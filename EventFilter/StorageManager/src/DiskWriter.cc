// $Id: DiskWriter.cc,v 1.18 2010/02/09 14:54:17 mommsen Exp $
/// @file: DiskWriter.cc

#include <algorithm>

#include <boost/bind.hpp>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/EventStreamHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


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

  // Destroy any remaining streams. Under normal conditions, there should be none
  destroyStreams(); 
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
    XCEPT_DECLARE_NESTED( stor::exception::DiskWriting,
                          sentinelException, errorMsg, e );
    _sharedResources->moveToFailedState(sentinelException);
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    XCEPT_DECLARE( stor::exception::DiskWriting,
                   sentinelException, errorMsg );
    _sharedResources->moveToFailedState(sentinelException);
  }
  catch(...)
  {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE( stor::exception::DiskWriting,
                   sentinelException, errorMsg );
    _sharedResources->moveToFailedState(sentinelException);
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
    _sharedResources->_diskWriterResources->setBusy(true);

    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addPoppedEventSample(event.memoryUsed());

    if( event.isEndOfLumiSectionMessage() )
      {
        processEndOfLumiSection( event );
      }
    else
      {
        writeEventToStreams( event );
        checkForFileTimeOuts();
      }

  }
  else
  {
    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);

    checkStreamChangeRequest();
    checkForFileTimeOuts(true);
    _sharedResources->_diskWriterResources->setBusy(false);
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


void DiskWriter::checkStreamChangeRequest()
{
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


void DiskWriter::checkForFileTimeOuts(const bool doItNow)
{
  utils::time_point_t now = utils::getCurrentTime();

  const DiskWritingParams dwParams =
    _sharedResources->_configuration->getDiskWritingParams();
  if (doItNow || (now - _lastFileTimeoutCheckTime) > dwParams._fileClosingTestInterval)
  {
    closeFilesForOldLumiSections();
    closeTimedOutFiles(now);
    _lastFileTimeoutCheckTime = now;
  }
}


void DiskWriter::closeFilesForOldLumiSections()
{
  uint32_t lumiSection;
  if (_sharedResources->_diskWriterResources->
    lumiSectionClosureRequested(lumiSection))
  {
    closeFilesForLumiSection(lumiSection);
  }
}


void DiskWriter::closeFilesForLumiSection(const uint32_t lumiSection)
{
  std::for_each(_streamHandlers.begin(), _streamHandlers.end(),
    boost::bind(&StreamHandler::closeFilesForLumiSection, _1, lumiSection));
}


void DiskWriter::closeTimedOutFiles(const utils::time_point_t now)
{
  std::for_each(_streamHandlers.begin(), _streamHandlers.end(),
    boost::bind(&StreamHandler::closeTimedOutFiles, _1, now));
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
    if ( it->fractionToDisk() > 0 )
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
  std::for_each(_streamHandlers.begin(), _streamHandlers.end(),
    boost::bind(&StreamHandler::closeAllFiles, _1));
  _streamHandlers.clear();
}


void DiskWriter::processEndOfLumiSection(const I2OChain& msg)
{
  const uint32_t ls = msg.lumiSection();
  std::for_each(_streamHandlers.begin(), _streamHandlers.end(),
    boost::bind(&StreamHandler::closeFilesForLumiSection, _1, ls));
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
