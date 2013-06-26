// $Id: DiskWriter.cc,v 1.33 2012/10/17 10:13:25 mommsen Exp $
/// @file: DiskWriter.cc

#include <algorithm>

#include <boost/bind.hpp>
#include <boost/pointer_cast.hpp>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/EventStreamHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FaultyEventStreamHandler.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {
  
  DiskWriter::DiskWriter(xdaq::Application *app, SharedResourcesPtr sr) :
  app_(app),
  sharedResources_(sr),
  dbFileHandler_(new DbFileHandler()),
  runNumber_(0),
  lastFileTimeoutCheckTime_(utils::getCurrentTime()),
  endOfRunReport_(new StreamsMonitorCollection::EndOfRunReport()),
  actionIsActive_(true)
  {
    WorkerThreadParams workerParams =
      sharedResources_->configuration_->getWorkerThreadParams();
    timeout_ = workerParams.DWdeqWaitTime_;
  }
  
  
  DiskWriter::~DiskWriter()
  {
    // Stop the activity
    actionIsActive_ = false;
    
    // Cancel the workloop (will wait until the action has finished)
    writingWL_->cancel();
    
    // Destroy any remaining streams. Under normal conditions, there should be none
    destroyStreams(); 
  }
  
  
  void DiskWriter::startWorkLoop(std::string workloopName)
  {
    try
    {
      std::string identifier = utils::getIdentifier(app_->getApplicationDescriptor());
      
      writingWL_ = toolbox::task::getWorkLoopFactory()->
        getWorkLoop( identifier + workloopName, "waiting" );
      
      if ( ! writingWL_->isActive() )
      {
        toolbox::task::ActionSignature* processAction = 
          toolbox::task::bind(this, &DiskWriter::writeAction, 
            identifier + "WriteNextEvent");
        writingWL_->submit(processAction);
        
        writingWL_->activate();
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
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    catch(std::exception &e)
    {
      errorMsg += e.what();
      XCEPT_DECLARE( stor::exception::DiskWriting,
        sentinelException, errorMsg );
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    catch(...)
    {
      errorMsg += "Unknown exception";
      XCEPT_DECLARE( stor::exception::DiskWriting,
        sentinelException, errorMsg );
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    
    return actionIsActive_;
  }
  
  
  void DiskWriter::writeNextEvent()
  {
    I2OChain event;
    StreamQueuePtr sq = sharedResources_->streamQueue_;
    utils::TimePoint_t startTime = utils::getCurrentTime();
    if (sq->deqTimedWait(event, timeout_))
    {
      sharedResources_->diskWriterResources_->setBusy(true);
      
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().addPoppedEventSample(event.memoryUsed());
      
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
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->
        getThroughputMonitorCollection().addDiskWriterIdleSample(elapsedTime);
      
      checkStreamChangeRequest();
      checkForFileTimeOuts(true);
      sharedResources_->diskWriterResources_->setBusy(false);
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
        streamHandlers_.at(*it)->writeEvent(event);
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
    if (sharedResources_->diskWriterResources_->
      streamChangeRequested(doConfig, evtCfgList, errCfgList, newdwParams, newRunNumber, newTimeoutValue))
    {
      destroyStreams();
      if (doConfig)
      {
        dwParams_ = newdwParams;
        runNumber_ = newRunNumber;
        timeout_ = newTimeoutValue;
        dbFileHandler_->configure(runNumber_, dwParams_);
        
        makeFaultyEventStream();
        configureEventStreams(evtCfgList);
        configureErrorStreams(errCfgList);
      }
      sharedResources_->diskWriterResources_->streamChangeDone();
    }
  }
  
  
  void DiskWriter::checkForFileTimeOuts(const bool doItNow)
  {
    utils::TimePoint_t now = utils::getCurrentTime();
    
    if (doItNow || (now - lastFileTimeoutCheckTime_) > dwParams_.fileClosingTestInterval_)
    {
      closeTimedOutFiles(now);
      lastFileTimeoutCheckTime_ = now;
    }
  }
  
  
  void DiskWriter::closeTimedOutFiles(const utils::TimePoint_t now)
  {
    std::for_each(streamHandlers_.begin(), streamHandlers_.end(),
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
    if ( dwParams_.faultyEventsStream_.empty() ) return;

    boost::shared_ptr<FaultyEventStreamHandler> newHandler(
    new FaultyEventStreamHandler(sharedResources_, dbFileHandler_, dwParams_.faultyEventsStream_)
    );
    streamHandlers_.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
  }
  
  
  void DiskWriter::makeEventStream(EventStreamConfigurationInfo& streamCfg)
  {
    boost::shared_ptr<EventStreamHandler> newHandler(
      new EventStreamHandler(streamCfg, sharedResources_, dbFileHandler_)
    );
    streamHandlers_.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
    streamCfg.setStreamId(streamHandlers_.size() - 1);
  }
  
  
  void DiskWriter::makeErrorStream(ErrorStreamConfigurationInfo& streamCfg)
  {
    boost::shared_ptr<FRDStreamHandler> newHandler(
      new FRDStreamHandler(streamCfg, sharedResources_, dbFileHandler_)
    );
    streamHandlers_.push_back(boost::dynamic_pointer_cast<StreamHandler>(newHandler));
    streamCfg.setStreamId(streamHandlers_.size() - 1);
  }
  
  
  void DiskWriter::destroyStreams()
  {
    if (streamHandlers_.empty()) return;
    
    std::for_each(streamHandlers_.begin(), streamHandlers_.end(),
      boost::bind(&StreamHandler::closeAllFiles, _1));
    streamHandlers_.clear();
    
    reportRemainingLumiSections();
    writeEndOfRunMarker();
  }
  
  
  void DiskWriter::reportRemainingLumiSections()
  {
    StreamsMonitorCollection& smc =
      sharedResources_->statisticsReporter_->getStreamsMonitorCollection();
    
    smc.reportAllLumiSectionInfos(dbFileHandler_, endOfRunReport_);
  }
  
  
  void DiskWriter::writeEndOfRunMarker()
  {
    std::ostringstream str;
    str << "LScount:" << endOfRunReport_->lsCountWithFiles
      << "\tEoLScount:" << endOfRunReport_->eolsCount
      << "\tLastLumi:" << endOfRunReport_->latestLumiSectionWritten
      << "\tEoR";
    dbFileHandler_->write(str.str());
    endOfRunReport_->reset();
  }
  
  
  void DiskWriter::processEndOfLumiSection(const I2OChain& msg)
  {
    if ( msg.faulty() || msg.runNumber() != runNumber_ ) return;
    if ( streamHandlers_.empty() ) return; //Don't care about EoLS signal if we have no streams

    const uint32_t lumiSection = msg.lumiSection();
    
    std::string fileCountStr;
    bool filesWritten = false;

    for (StreamHandlers::const_iterator it = streamHandlers_.begin(),
           itEnd = streamHandlers_.end(); it != itEnd; ++it)
    {
      if ( (*it)->closeFilesForLumiSection(lumiSection, fileCountStr) )
        filesWritten = true;
    }
    fileCountStr += "\tEoLS:1";
    dbFileHandler_->write(fileCountStr);

    ++(endOfRunReport_->eolsCount);
    if (filesWritten) ++(endOfRunReport_->lsCountWithFiles);
    endOfRunReport_->updateLatestWrittenLumiSection(lumiSection);
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
