// $Id: StateMachine.cc,v 1.4 2011/05/09 11:03:34 mommsen Exp $
/// @file: StateMachine.cc

#include "EventFilter/SMProxyServer/interface/DataManager.h"
#include "EventFilter/SMProxyServer/interface/StateMachine.h"
#include "EventFilter/SMProxyServer/interface/States.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"

#include "xdata/InfoSpace.h"

#include <boost/bind.hpp>

#include <sstream>


namespace smproxy
{
  StateMachine::StateMachine
  (
    xdaq::Application* app
  ):
  app_(app),
  rcmsStateNotifier_
  (
    app->getApplicationLogger(),
    app->getApplicationDescriptor(),
    app->getApplicationContext()
  ),
  reasonForFailed_(""),
  stateName_("Halted")
  {
    std::ostringstream oss;
    oss << app->getApplicationDescriptor()->getClassName()
      << app->getApplicationDescriptor()->getInstance();
    appNameAndInstance_ = oss.str();

    xdata::InfoSpace *is = app->getApplicationInfoSpace();
    is->fireItemAvailable("rcmsStateListener",
      rcmsStateNotifier_.getRcmsStateListenerParameter() );
    is->fireItemAvailable("foundRcmsStateListener",
      rcmsStateNotifier_.getFoundRcmsStateListenerParameter() );
    rcmsStateNotifier_.findRcmsStateListener();
    rcmsStateNotifier_.subscribeToChangesInRcmsStateListener(is);
  
    is->fireItemAvailable("stateName", &stateName_);

    initiate();

    configuration_.reset(new Configuration(
        app->getApplicationInfoSpace(), app->getApplicationDescriptor()->getInstance()
      ));

    registrationCollection_.reset( new stor::RegistrationCollection() );
    
    registrationQueue_.reset(new stor::RegistrationQueue(
        configuration_->getQueueConfigurationParams().registrationQueueSize_
      ));
    
    initMsgCollection_.reset(new stor::InitMsgCollection());

    statisticsReporter_.reset(new StatisticsReporter(app,
        configuration_->getQueueConfigurationParams()));

    eventQueueCollection_.reset(new EventQueueCollection(
        statisticsReporter_->getEventConsumerMonitorCollection()));
    
    dqmEventQueueCollection_.reset(new stor::DQMEventQueueCollection(
        statisticsReporter_->getDQMConsumerMonitorCollection()));
    
    dataManager_.reset(new DataManager(this));
  }
  
  
  std::string StateMachine::processEvent(const boost::statechart::event_base& event)
  {
    boost::mutex::scoped_lock sl(eventMutex_);
    process_event(event);
    return state_cast<const StateName&>().stateName();
  }
  
  
  void StateMachine::setExternallyVisibleStateName(const std::string& stateName)
  {
    stateName_ = stateName;
    rcmsStateNotifier_.stateChanged(stateName,
      appNameAndInstance_ + " has reached target state " +
      stateName);
  }
  
  
  void StateMachine::failEvent(const Fail& evt)
  {
    stateName_ = "Failed";
    reasonForFailed_ = evt.getTraceback();
    
    LOG4CPLUS_FATAL(app_->getApplicationLogger(),
      "Failed: " << evt.getReason() << ". " << reasonForFailed_);
    
    rcmsStateNotifier_.stateChanged(stateName_, evt.getReason());
    
    app_->notifyQualified("fatal", evt.getException());
  }
  
  
  void StateMachine::unconsumed_event(const boost::statechart::event_base& evt)
  {
    LOG4CPLUS_ERROR(app_->getApplicationLogger(),
      "The " << typeid(evt).name()
      << " event is not supported from the "
      << stateName_.toString() << " state!");
  }
  
  
  void StateMachine::updateConfiguration()
  {
    std::string errorMsg = "Failed to update configuration parameters";
    try
    {
      configuration_->updateAllParams();
    }
    catch(xcept::Exception &e)
    {
      XCEPT_DECLARE_NESTED(exception::Configuration,
        sentinelException, errorMsg, e);
      moveToFailedState(sentinelException);
    }
    catch( std::exception &e )
    {
      errorMsg.append(": ");
      errorMsg.append( e.what() );
      
      XCEPT_DECLARE(exception::Configuration,
        sentinelException, errorMsg);
      moveToFailedState(sentinelException);
    }
    catch(...)
    {
      errorMsg.append(": unknown exception");
      
      XCEPT_DECLARE(exception::Configuration,
        sentinelException, errorMsg);
      moveToFailedState(sentinelException);
    }
  }
  
  
  void StateMachine::setQueueSizes()
  {
    QueueConfigurationParams queueParams =
      configuration_->getQueueConfigurationParams();
    registrationQueue_->
      setCapacity(queueParams.registrationQueueSize_);
  }
  
  
  void StateMachine::setAlarms()
  {
    AlarmParams alarmParams =
      configuration_->getAlarmParams();
    statisticsReporter_->getDataRetrieverMonitorCollection().
      configureAlarms(alarmParams);
  }
  
  
  void StateMachine::clearInitMsgCollection()
  {
    initMsgCollection_->clear();
  }
  
  
  void StateMachine::resetStatistics()
  {
    statisticsReporter_->reset();
  }
  
  
  void StateMachine::clearConsumerRegistrations()
  {
    registrationCollection_->clearRegistrations();
    eventQueueCollection_->removeQueues();
    dqmEventQueueCollection_->removeQueues();
  }
  
  
  void StateMachine::enableConsumerRegistration()
  {
    registrationCollection_->enableConsumerRegistration();
    dataManager_->start(configuration_->getDataRetrieverParams());
  }
 
  
  void StateMachine::disableConsumerRegistration()
  {
    registrationCollection_->disableConsumerRegistration();
    dataManager_->stop();
  }
 
  
  void StateMachine::clearQueues()
  {
    registrationQueue_->clear();
    eventQueueCollection_->clearQueues();
    dqmEventQueueCollection_->clearQueues();
  }
  
  
  void Configuring::entryAction()
  {
    configuringThread_.reset(
      new boost::thread( boost::bind( &Configuring::activity, this) )
    );
  }
  
  
  void Configuring::activity()
  {
    outermost_context_type& stateMachine = outermost_context();
    stateMachine.updateConfiguration();
    boost::this_thread::interruption_point();
    stateMachine.setQueueSizes();
    boost::this_thread::interruption_point();
    stateMachine.setAlarms();
    boost::this_thread::interruption_point();
    stateMachine.processEvent( ConfiguringDone() );
  }

  
  void Configuring::exitAction()
  {
    configuringThread_->interrupt();
  }
  
  
  void Starting::entryAction()
  {
    startingThread_.reset(
      new boost::thread( boost::bind( &Starting::activity, this) )
    );
  }
  
  
  void Starting::activity()
  {
    outermost_context_type& stateMachine = outermost_context();
    stateMachine.clearInitMsgCollection();
    boost::this_thread::interruption_point();
    stateMachine.resetStatistics();
    boost::this_thread::interruption_point();
    stateMachine.clearConsumerRegistrations();
    boost::this_thread::interruption_point();
    stateMachine.enableConsumerRegistration();
    boost::this_thread::interruption_point();
    stateMachine.processEvent( StartingDone() );
  }

  
  void Starting::exitAction()
  {
    startingThread_->interrupt();
  }
  
  
  void Stopping::entryAction()
  {
    stoppingThread_.reset(
      new boost::thread( boost::bind( &Stopping::activity, this) )
    );
  }
  
  
  void Stopping::activity()
  {
    outermost_context_type& stateMachine = outermost_context();
    stateMachine.disableConsumerRegistration();
    boost::this_thread::interruption_point();
    stateMachine.clearQueues();
    boost::this_thread::interruption_point();
    stateMachine.processEvent( StoppingDone() );
  }

  
  void Stopping::exitAction()
  {
    stoppingThread_->interrupt();
  }
  
  
  void Halting::entryAction()
  {
    haltingThread_.reset(
      new boost::thread( boost::bind( &Halting::activity, this) )
    );
  }
  
  
  void Halting::activity()
  {
    outermost_context_type& stateMachine = outermost_context();
    stateMachine.disableConsumerRegistration();
    boost::this_thread::interruption_point();
    stateMachine.clearQueues();
    boost::this_thread::interruption_point();
    stateMachine.processEvent( HaltingDone() );
  }

  
  void Halting::exitAction()
  {
    haltingThread_->interrupt();
  }

} // namespace smproxy


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
