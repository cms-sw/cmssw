// $Id: FragmentProcessor.cc,v 1.15 2010/05/11 15:18:28 mommsen Exp $
/// @file: FragmentProcessor.cc

#include <unistd.h>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentProcessor.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"

using namespace stor;


FragmentProcessor::FragmentProcessor( xdaq::Application *app,
                                      SharedResourcesPtr sr ) :
  _app(app),
  _sharedResources(sr),
  _wrapperNotifier( app ),
  _fragmentStore(),
  _eventDistributor(sr),
  _actionIsActive(true)
{
  _stateMachine.reset( new StateMachine( &_eventDistributor,
                                         &_fragmentStore, &_wrapperNotifier,
                                         _sharedResources ) );
  _stateMachine->initiate();

  WorkerThreadParams workerParams =
    _sharedResources->_configuration->getWorkerThreadParams();
  _timeout = workerParams._FPdeqWaitTime;
}

FragmentProcessor::~FragmentProcessor()
{
  // Stop the activity
  _actionIsActive = false;

  // Cancel the workloop (will wait until the action has finished)
  _processWL->cancel();
}

void FragmentProcessor::startWorkLoop(std::string workloopName)
{
  try
    {
      std::string identifier = utils::getIdentifier(_app->getApplicationDescriptor());

      _processWL = toolbox::task::getWorkLoopFactory()->
        getWorkLoop( identifier + workloopName, "waiting" );

      if ( ! _processWL->isActive() )
        {
          toolbox::task::ActionSignature* processAction = 
            toolbox::task::bind(this, &FragmentProcessor::processMessages, 
                                identifier + "ProcessMessages");
          _processWL->submit(processAction);

          _processWL->activate();
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
    _sharedResources->_statisticsReporter->alarmHandler()->
      notifySentinel(AlarmHandler::ERROR, e);
  }
  catch(xcept::Exception &e)
  {
    XCEPT_DECLARE_NESTED( stor::exception::FragmentProcessing,
                          sentinelException, errorMsg, e );
    _sharedResources->moveToFailedState(sentinelException);
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState(sentinelException);
  }
  catch(...)
  {
    errorMsg += "Unknown exception";
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    _sharedResources->moveToFailedState(sentinelException);
  }

  return _actionIsActive;
}

void FragmentProcessor::processOneFragmentIfPossible()
{
  if (_eventDistributor.full()) 
  {
    utils::time_point_t startTime = utils::getCurrentTime();

    ::usleep(_timeout.total_microseconds());

    utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
    _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addFragmentProcessorIdleSample(elapsedTime);

    _fragmentStore.addToStaleEventTimes(elapsedTime);
  }
  else 
    processOneFragment();
}

void FragmentProcessor::processOneFragment()
{
  I2OChain fragment;
  boost::shared_ptr<FragmentQueue> fq = _sharedResources->_fragmentQueue;
  utils::time_point_t startTime = utils::getCurrentTime();
  if (fq->deq_timed_wait(fragment, _timeout))
    {
      utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
      _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addFragmentProcessorIdleSample(elapsedTime);
      _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addPoppedFragmentSample(fragment.memoryUsed());

      _stateMachine->getCurrentState().processI2OFragment(fragment);
    }
  else
    {
      utils::duration_t elapsedTime = utils::getCurrentTime() - startTime;
      _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addFragmentProcessorIdleSample(elapsedTime);

      _stateMachine->getCurrentState().noFragmentToProcess();  
    }
}


void FragmentProcessor::processAllCommands()
{
  boost::shared_ptr<CommandQueue> cq = _sharedResources->_commandQueue;
  stor::event_ptr evt;
  bool gotCommand = false;

  while( cq->deq_nowait( evt ) )
    {
      gotCommand = true;
      _stateMachine->process_event( *evt );
    }

  // the timeout value may have changed if the transition was
  // a Configuration transition, so check for a new value here
  if (gotCommand)
    {
      WorkerThreadParams workerParams =
        _sharedResources->_configuration->getWorkerThreadParams();
      _timeout = workerParams._FPdeqWaitTime;
    }
}


void FragmentProcessor::processAllRegistrations()
{
  RegInfoBasePtr regInfo;
  boost::shared_ptr<RegistrationQueue> regQueue =
    _sharedResources->_registrationQueue;
  while ( regQueue->deq_nowait( regInfo ) )
    {
      regInfo->registerMe( &_eventDistributor );
    }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
