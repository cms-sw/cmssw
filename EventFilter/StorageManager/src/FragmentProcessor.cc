// $Id: FragmentProcessor.cc,v 1.7 2009/06/29 15:47:29 mommsen Exp $

#include <unistd.h>

#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentProcessor.h"

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
  _timeout = (unsigned int) workerParams._FPdeqWaitTime;
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
  catch(stor::exception::RunNumberMismatch &e)
  {
    LOG4CPLUS_ERROR(_app->getApplicationLogger(), e.message());

    _app->notifyQualified("error", e);
  }
  catch(xcept::Exception &e)
  {
    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg << xcept::stdformat_exception_history(e));

    XCEPT_DECLARE_NESTED(stor::exception::FragmentProcessing,
      sentinelException, errorMsg, e);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState();
  }
  catch(std::exception &e)
  {
    errorMsg += e.what();

    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState();
  }
  catch(...)
  {
    errorMsg += "Unknown exception";

    LOG4CPLUS_FATAL(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::FragmentProcessing,
      sentinelException, errorMsg);
    _app->notifyQualified("fatal", sentinelException);

    _sharedResources->moveToFailedState();
  }

  return _actionIsActive;
}

void FragmentProcessor::processOneFragmentIfPossible()
{
  if (_eventDistributor.full()) 
  {
    utils::time_point_t startTime = utils::getCurrentTime();

    // 27-May-2009, KAB - This is rather ugly. At the moment, we are limited
    // to wait times of an integer number of seconds by the deq_timed_wait
    // call below. (The Configuration class enforces this rule.)  Once we
    // can support sub-second wait times [Boost 1.38?], we can relax this
    // rule.  In the meantime, the ThroughputMonitorCollection busy time
    // calculation for this thread gives smoother results if we don't sleep
    // for full seconds here.  
    if (_timeout > 0.1)
    {
      ::usleep(100000);
    }
    else
    {
      ::sleep(_timeout);
    }

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
      _sharedResources->_statisticsReporter->getThroughputMonitorCollection().addPoppedFragmentSample(fragment.totalDataSize());

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
      _timeout = (unsigned int) workerParams._FPdeqWaitTime;
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
