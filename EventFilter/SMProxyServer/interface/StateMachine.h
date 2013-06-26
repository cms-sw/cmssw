// $Id: StateMachine.h,v 1.5 2011/05/09 11:03:25 mommsen Exp $
/// @file: StateMachine.h 

#ifndef EventFilter_SMProxyServer_StateMachine_h
#define EventFilter_SMProxyServer_StateMachine_h

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/DataManager.h"
#include "EventFilter/SMProxyServer/interface/EventQueueCollection.h"
#include "EventFilter/SMProxyServer/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationQueue.h"

#include "xcept/Exception.h"
#include "xcept/tools.h"
#include "xdaq/Application.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdaq2rc/RcmsStateNotifier.h"

#include <boost/shared_ptr.hpp>
#include <boost/statechart/event_base.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/state_machine.hpp>
#include <boost/thread/mutex.hpp>

#include <string>


namespace smproxy
{

  //////////////////////
  // Outermost states //
  //////////////////////
  class AllOk;
  class Failed;

  ///////////////////
  // Public events //
  ///////////////////
  class Configure : public boost::statechart::event<Configure> {};
  class Enable : public boost::statechart::event<Enable> {};
  class Halt : public boost::statechart::event<Halt> {};
  class Stop : public boost::statechart::event<Stop> {};

  class Fail : public boost::statechart::event<Fail>
  {
  public:
    Fail(xcept::Exception& exception) : exception_(exception) {};
    std::string getReason() const { return exception_.message(); }
    std::string getTraceback() const { return xcept::stdformat_exception_history(exception_); }
    xcept::Exception& getException() const { return exception_; }

  private:
    mutable xcept::Exception exception_;
  };

  struct StateName {
    virtual std::string stateName() const = 0;
  };

  ///////////////////////
  // The state machine //
  ///////////////////////

  class StateMachine: public boost::statechart::state_machine<StateMachine,AllOk>
  {
    
  public:
    
    StateMachine
    (
      xdaq::Application*
    );

    std::string processEvent(const boost::statechart::event_base&);
    void moveToFailedState(xcept::Exception& e)
    { processEvent( Fail(e) ); }

    void setExternallyVisibleStateName(const std::string& stateName);
    void failEvent(const Fail&);
    void unconsumed_event(const boost::statechart::event_base&);

    std::string getReasonForFailed()
    { return reasonForFailed_; }
    std::string getStateName()
    { return state_cast<const StateName&>().stateName(); }
    std::string getExternallyVisibleStateName()
    { return stateName_.toString(); }

    ConfigurationPtr getConfiguration() const
    { return configuration_; }
    DataManagerPtr getDataManager() const
    { return dataManager_; }
    stor::RegistrationCollectionPtr getRegistrationCollection() const
    { return registrationCollection_; }
    stor::RegistrationQueuePtr getRegistrationQueue() const
    { return registrationQueue_; }
    stor::InitMsgCollectionPtr getInitMsgCollection() const
    { return initMsgCollection_; }
    EventQueueCollectionPtr getEventQueueCollection() const
    { return eventQueueCollection_; }
    stor::DQMEventQueueCollectionPtr getDQMEventQueueCollection() const
    { return dqmEventQueueCollection_; }
    StatisticsReporterPtr getStatisticsReporter() const
    { return statisticsReporter_; }
    xdaq::ApplicationDescriptor* getApplicationDescriptor() const
    { return app_->getApplicationDescriptor(); }

    void updateConfiguration();
    void setQueueSizes();
    void setAlarms();
    void clearInitMsgCollection();
    void resetStatistics();
    void clearConsumerRegistrations();
    void enableConsumerRegistration();
    void disableConsumerRegistration();
    void clearQueues();
    
    
  private:

    xdaq::Application* app_;
    xdaq2rc::RcmsStateNotifier rcmsStateNotifier_;
    ConfigurationPtr configuration_;
    DataManagerPtr dataManager_;
    stor::RegistrationCollectionPtr registrationCollection_;
    stor::RegistrationQueuePtr registrationQueue_;
    stor::InitMsgCollectionPtr initMsgCollection_;
    StatisticsReporterPtr statisticsReporter_;
    EventQueueCollectionPtr eventQueueCollection_;
    stor::DQMEventQueueCollectionPtr dqmEventQueueCollection_;

    mutable boost::mutex eventMutex_;
    
    std::string appNameAndInstance_;
    std::string reasonForFailed_;
    xdata::String stateName_;
    
  };
  
  typedef boost::shared_ptr<StateMachine> StateMachinePtr;

} //namespace smproxy

#endif //SMProxyServer_StateMachine_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
