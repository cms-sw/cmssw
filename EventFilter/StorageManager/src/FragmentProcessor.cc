// $Id: FragmentProcessor.cc,v 1.20 2011/11/08 10:48:40 mommsen Exp $
/// @file: FragmentProcessor.cc

#include <unistd.h>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentProcessor.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"

using namespace stor;


FragmentProcessor::FragmentProcessor( xdaq::Application *app,
                                      SharedResourcesPtr sr ) :
  app_(app),
  sharedResources_(sr),
  wrapperNotifier_( app ),
  fragmentStore_(sr->configuration_->getQueueConfigurationParams().fragmentStoreMemoryLimitMB_),
  eventDistributor_(sr),
  actionIsActive_(true)
{
  stateMachine_.reset( new StateMachine( &eventDistributor_,
                                         &fragmentStore_, &wrapperNotifier_,
                                         sharedResources_ ) );
  stateMachine_->initiate();

  WorkerThreadParams workerParams =
    sharedResources_->configuration_->getWorkerThreadParams();
  timeout_ = workerParams.FPdeqWaitTime_;
}

FragmentProcessor::~FragmentProcessor()
{
  // Stop the activity
  actionIsActive_ = false;

  // Cancel the workloop (will wait until the action has finished)
  processWL_->cancel();
}

void FragmentProcessor::startWorkLoop(std::string workloopName)
{
  try
    {
      std::string identifier = utils::getIdentifier(app_->getApplicationDescriptor());

      processWL_ = toolbox::task::getWorkLoopFactory()->
        getWorkLoop( identifier + workloopName, "waiting" );

      if ( ! processWL_->isActive() )
        {
          toolbox::task::ActionSignature* processAction = 
            toolbox::task::bind(this, &FragmentProcessor::processMessages, 
                                identifier + "ProcessMessages");
          processWL_->submit(processAction);

          processWL_->activate();
        }
    }
  catch (xcept::Exception& e)
    {
      std::string msg = "Failed to start workloop 'FragmentProcessor' with 'processMessages'.";
    XCEPT_RETHROW(stor::exception::FragmentProcessing, msg, e);
  }
}

bool FragmentProcessor::processMessages(toolbox::task::WorkLoop*)
{
  std::string errorMsg;

  try
  {
    errorMsg = "Failed to process state machine events: ";
    processAllCommands();
    
    errorMsg = "Failed to process consumer registrations: ";
    processAllRegistrations();
    
    errorMsg = "Failed to process an event fragment: ";
    processOneFragmentIfPossible();
  }
  catch(stor::exception::RBLookupFailed &e)
  {
    sharedResources_->alarmHandler_->
      notifySentinel(AlarmHandler::ERROR, e);
  }
  catch(xcept::Exception &e)
  {
    XCEPT_DECLARE_NESTED( stor::exception::FragmentProcessing,
                          sentinelException, errorMsg, e );
    sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
  }
  catch(...)
  {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    sharedResources_->alarmHandler_->moveToFailedState(sentinelException);
  }

  return actionIsActive_;
}

void FragmentProcessor::processOneFragmentIfPossible()
{
  if (fragmentStore_.full() || eventDistributor_.full()) 
  {
    utils::TimePoint_t startTime = utils::getCurrentTime();

    utils::sleep(timeout_);

    utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
    sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
      addFragmentProcessorIdleSample(elapsedTime);

    fragmentStore_.addToStaleEventTimes(elapsedTime);
  }
  else 
    processOneFragment();
}

void FragmentProcessor::processOneFragment()
{
  I2OChain fragment;
  FragmentQueuePtr fq = sharedResources_->fragmentQueue_;
  utils::TimePoint_t startTime = utils::getCurrentTime();
  if (fq->deqTimedWait(fragment, timeout_))
    {
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
        addFragmentProcessorIdleSample(elapsedTime);
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
        addPoppedFragmentSample(fragment.memoryUsed());

      stateMachine_->getCurrentState().processI2OFragment(fragment);
    }
  else
    {
      utils::Duration_t elapsedTime = utils::getCurrentTime() - startTime;
      sharedResources_->statisticsReporter_->getThroughputMonitorCollection().
        addFragmentProcessorIdleSample(elapsedTime);

      stateMachine_->getCurrentState().noFragmentToProcess();  
    }
}


void FragmentProcessor::processAllCommands()
{
  CommandQueuePtr cq = sharedResources_->commandQueue_;
  stor::EventPtr_t evt;
  bool gotCommand = false;

  while( cq->deqNowait( evt ) )
    {
      gotCommand = true;
      stateMachine_->process_event( *evt );
    }

  // the timeout value may have changed if the transition was
  // a Configuration transition, so check for a new value here
  if (gotCommand)
    {
      WorkerThreadParams workerParams =
        sharedResources_->configuration_->getWorkerThreadParams();
      timeout_ = workerParams.FPdeqWaitTime_;
    }
}


void FragmentProcessor::processAllRegistrations()
{
  RegPtr regPtr;
  RegistrationQueuePtr regQueue =
    sharedResources_->registrationQueue_;
  while ( regQueue->deqNowait( regPtr ) )
  {
    regPtr->registerMe( &eventDistributor_ );
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
