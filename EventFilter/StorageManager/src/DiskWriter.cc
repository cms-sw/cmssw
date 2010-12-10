// $Id: DiskWriter.cc,v 1.24 2010/10/12 01:41:32 wmtan Exp $
/// @file: DiskWriter.cc

#include <algorithm>

#include <boost/bind.hpp>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/EventStreamHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FaultyEventStreamHandler.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


using namespace stor;

DiskWriter::DiskWriter(xdaq::Application *app, SharedResourcesPtr sr) :
_app(app),
_sharedResources(sr),
_dbFileHandler(new DbFileHandler()),
_runNumber(0),
_lastFileTimeoutCheckTime(utils::getCurrentTime()),
_actionIsActive(true)
{
  WorkerThreadParams workerParams =
    _sharedResources->_configuration->getWorkerThreadParams();
  _timeout = workerParams._DWdeqWaitTime;
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
    _sharedResources->_statisticsReporter->
      getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);

    checkStreamChangeRequest();
    checkForFileTimeOuts(true);
    _sharedResources->_diskWriterResources->setBusy(false);
  }
}


void DiskWriter::writeEventToStreams(const I2OChain& event)
{
  std::vector<StreamID> streams = event.getStreamTags();

  for (
    std::vector<StreamID>::const_iterator it = streams.begin(), itEnd = streams.end();
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
  DiskWritingParams newdwParams;
  unsigned int newRunNumber;
  boost::posix_time::time_duration newTimeoutValue;
  bool doConfig;
  if (_sharedResources->_diskWriterResources->
    streamChangeRequested(doConfig, evtCfgList, errCfgList, newdwParams, newRunNumber, newTimeoutValue))
  {
    destroyStreams();
    if (doConfig)
    {
      _dwParams = newdwParams;
      _runNumber = newRunNumber;
      _timeout = newTimeoutValue;
      _dbFileHandler->configure(_runNumber, _dwParams);

      makeFaultyEventStream();
      configureEventStreams(evtCfgList);
      configureErrorStreams(errCfgList);
    }
    _sharedResources->_diskWriterResources->streamChangeDone();
  }
}


void DiskWriter::checkForFileTimeOuts(const bool doItNow)
{
  utils::time_point_t now = utils::getCurrentTime();

  if (doItNow || (now - _lastFileTimeoutCheckTime) > _dwParams._fileClosingTestInterval)
  {
    closeTimedOutFiles(now);
    _lastFileTimeoutCheckTime = now;
  }
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


void DiskWriter::makeFaultyEventStream()
{
  if ( _dwParams._faultyEventsStream.empty() ) return;

  boost::shared_ptr<FaultyEventStreamHandler> newHandler(
    new FaultyEventStreamHandler(_sharedResources, _dbFileHandler, _dwParams._faultyEventsStream)
  );
  _streamHandlers.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
}


void DiskWriter::makeEventStream(EventStreamConfigurationInfo& streamCfg)
{
  boost::shared_ptr<EventStreamHandler> newHandler(
    new EventStreamHandler(streamCfg, _sharedResources, _dbFileHandler)
  );
  _streamHandlers.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
  streamCfg.setStreamId(_streamHandlers.size() - 1);
}


void DiskWriter::makeErrorStream(ErrorStreamConfigurationInfo& streamCfg)
{
  boost::shared_ptr<FRDStreamHandler> newHandler(
    new FRDStreamHandler(streamCfg, _sharedResources, _dbFileHandler)
  );
  _streamHandlers.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
  streamCfg.setStreamId(_streamHandlers.size() - 1);
}


void DiskWriter::destroyStreams()
{
  if (_streamHandlers.empty()) return;

  std::for_each(_streamHandlers.begin(), _streamHandlers.end(),
    boost::bind(&StreamHandler::closeAllFiles, _1));
  _streamHandlers.clear();
  
  reportRemainingLumiSections();
  writeEndOfRunMarker();
}


void DiskWriter::reportRemainingLumiSections() const
{
  StreamsMonitorCollection& smc =
    _sharedResources->_statisticsReporter->getStreamsMonitorCollection();
  
  smc.reportAllLumiSectionInfos(_dbFileHandler);
}


void DiskWriter::writeEndOfRunMarker() const
{
  RunMonitorCollection& rmc =
    _sharedResources->_statisticsReporter->getRunMonitorCollection();
  // Make sure we report the latest values
  rmc.calculateStatistics(utils::getCurrentTime());

  MonitoredQuantity::Stats lumiSectionsSeenStats;
  rmc.getLumiSectionsSeenMQ().getStats(lumiSectionsSeenStats);
  MonitoredQuantity::Stats eolsSeenStats;
  rmc.getEoLSSeenMQ().getStats(eolsSeenStats);

  std::ostringstream str;
  str << "LScount:" << lumiSectionsSeenStats.getSampleCount()
    << "\tEoLScount:" << eolsSeenStats.getSampleCount()
    << "\tLastLumi:" << lumiSectionsSeenStats.getLastSampleValue()
    << "\tEoR";
  _dbFileHandler->write(str.str());
}


void DiskWriter::processEndOfLumiSection(const I2OChain& msg)
{
  if ( msg.faulty() || msg.runNumber() != _runNumber ) return;

  const uint32_t lumiSection = msg.lumiSection();

  std::string fileCountStr;

  for (StreamHandlers::const_iterator it = _streamHandlers.begin(),
         itEnd = _streamHandlers.end(); it != itEnd; ++it)
  {
    (*it)->closeFilesForLumiSection(lumiSection, fileCountStr);
  }
  _dbFileHandler->write(fileCountStr);
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
