// $Id: DQMEventProcessor.cc,v 1.19 2012/04/20 10:48:01 mommsen Exp $
/// @file: DQMEventProcessor.cc

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessor.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/src/DQMEventStore.icc"


namespace stor {
  
  ///////////////////////////////////////
  // Specializations for DQMEventStore //
  ///////////////////////////////////////
  
  template<>  
  DQMEventMsgView
  DQMEventStore<I2OChain,DataSenderMonitorCollection,AlarmHandler>::
  getDQMEventView(I2OChain const& dqmEvent)
  {
    tempEventArea_.clear();
    dqmEvent.copyFragmentsIntoBuffer(tempEventArea_);
    return DQMEventMsgView(&tempEventArea_[0]);
  }



  DQMEventProcessor::DQMEventProcessor(xdaq::Application* app, SharedResourcesPtr sr) :
  app_(app),
  sharedResources_(sr),
  actionIsActive_(true),
  latestLumiSection_(0),
  discardDQMUpdatesForOlderLS_(0),
  dqmEventStore_
  (
    app->getApplicationDescriptor(),
    sr->dqmEventQueueCollection_,
    sr->statisticsReporter_->getDQMEventMonitorCollection(),
    &sr->statisticsReporter_->getDataSenderMonitorCollection(),
    &stor::DataSenderMonitorCollection::getConnectedEPs,
    sr->alarmHandler_.get(),
    &stor::AlarmHandler::moveToFailedState,
    sr->alarmHandler_
  )
  {
    WorkerThreadParams workerParams =
      sharedResources_->configuration_->getWorkerThreadParams();
    timeout_ = workerParams.DQMEPdeqWaitTime_;
  }
  
  
  DQMEventProcessor::~DQMEventProcessor()
  {
    // Stop the activity
    actionIsActive_ = false;
    
    // Cancel the workloop (will wait until the action has finished)
    processWL_->cancel();
  }
  
  
  void DQMEventProcessor::startWorkLoop(std::string workloopName)
  {
    try
    {
      std::string identifier = utils::getIdentifier(app_->getApplicationDescriptor());
      
      processWL_ = toolbox::task::getWorkLoopFactory()->
      getWorkLoop( identifier + workloopName, "waiting" );
      
      if ( ! processWL_->isActive() )
      {
        toolbox::task::ActionSignature* processAction = 
          toolbox::task::bind(this, &DQMEventProcessor::processDQMEvents,
          identifier + "ProcessNextDQMEvent");
        processWL_->submit(processAction);
        
        processWL_->activate();
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
      XCEPT_DECLARE_NESTED( stor::exception::DQMEventProcessing,
        sentinelException, errorMsg, e );
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    catch(std::exception &e)
    {
      errorMsg += e.what();
      XCEPT_DECLARE( stor::exception::DQMEventProcessing,
        sentinelException, errorMsg );
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    catch(...)
    {
      errorMsg += "Unknown exception";
      XCEPT_DECLARE( stor::exception::DQMEventProcessing,
        sentinelException, errorMsg );
      sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
    }
    
    return actionIsActive_;
  }
  
  
  void DQMEventProcessor::processNextDQMEvent()
  {
    DQMEventQueue::ValueType dqmEvent;
    DQMEventQueuePtr eq = sharedResources_->dqmEventQueue_;
    utils::TimePoint_t startTime = utils::getCurrentTime();
    if (eq->deqTimedWait(dqmEvent, timeout_))
    {
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
        addDQMEventProcessorIdleSample(elapsedTime);

      if (
        (discardDQMUpdatesForOlderLS_ > 0) &&
        (dqmEvent.first.lumiSection() + discardDQMUpdatesForOlderLS_ < latestLumiSection_)
      )
        // subtracting unsigned quantities might not yield the right result!
      {
        // discard very old LS
        sharedResources_->statisticsReporter_->getDQMEventMonitorCollection().
          getDroppedDQMEventCountsMQ().addSample(dqmEvent.second + 1);        
      }
      else
      {
        sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
          addPoppedDQMEventSample(dqmEvent.first.memoryUsed());
        sharedResources_->statisticsReporter_->getDQMEventMonitorCollection().
          getDroppedDQMEventCountsMQ().addSample(dqmEvent.second);
        
        latestLumiSection_ = std::max(latestLumiSection_, dqmEvent.first.lumiSection());
        dqmEventStore_.addDQMEvent(dqmEvent.first);
      }
    }
    else
    {
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
        addDQMEventProcessorIdleSample(elapsedTime);
    }
    
    DQMEventProcessorResources::Requests requests;
    DQMProcessingParams dqmParams;
    boost::posix_time::time_duration newTimeoutValue;
    if (sharedResources_->dqmEventProcessorResources_->
      getRequests(requests, dqmParams, newTimeoutValue))
    {
      if (requests.configuration)
      {
        timeout_ = newTimeoutValue;
        dqmEventStore_.setParameters(dqmParams);
        discardDQMUpdatesForOlderLS_ = dqmParams.discardDQMUpdatesForOlderLS_;
      }
      if (requests.endOfRun)
      {
        endOfRun();
      }
      if (requests.storeDestruction)
      {
        dqmEventStore_.clear();
      }
      sharedResources_->dqmEventProcessorResources_->requestsDone();
    }
  }
  
  void DQMEventProcessor::endOfRun()
  {
    dqmEventStore_.purge();
    latestLumiSection_ = 0;
  }
    
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
