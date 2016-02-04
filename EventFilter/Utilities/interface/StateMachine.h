#ifndef EVF_STATEMACHINE_H
#define EVF_STATEMACHINE_H 1


#include "xdaq/Application.h"
#include "xdaq/NamespaceURI.h"

#include "toolbox/fsm/FiniteStateMachine.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "toolbox/task/Action.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"

#include "xdata/String.h"
#include "xdata/Bag.h"

#include "xdaq2rc/ClassnameAndInstance.h"
#include "xdaq2rc/RcmsStateNotifier.h"

#include <string>


namespace evf
{
  
  class StateMachine : public toolbox::lang::Class
  {
  public:
    //
    // construction / destruction
    //
    StateMachine(xdaq::Application* app);
    virtual ~StateMachine();
    
    
    //
    // member functions 
    //
    
    // finite state machine command callback
    xoap::MessageReference commandCallback(xoap::MessageReference msg)
      throw (xoap::exception::Exception);
    
    // finite state machine callback for entering new state
    void stateChanged(toolbox::fsm::FiniteStateMachine & fsm) 
      throw (toolbox::fsm::exception::Exception);
    
    // finite state machine callback for transition into 'Failed' state
    void failed(toolbox::Event::Reference e)
      throw(toolbox::fsm::exception::Exception);
    
    // fire state transition event
    void fireEvent(const std::string& evtType,void* originator);
    
    // initiate transition to state 'Failed'
    void fireFailed(const std::string& errorMsg,void* originator);
    
    // find RCMS state listener
    void findRcmsStateListener();

    // disable rcms state notification
    void disableRcmsStateNotification(){doStateNotification_=false;}

    bool checkIfEnabled() {return fsm_.getCurrentState()=='E';}
    
    // report current state
    xdata::String* stateName() { return &stateName_; }
    
    // get the RCMS StateListener parameter (classname/instance)
    xdata::Bag<xdaq2rc::ClassnameAndInstance>* rcmsStateListener()
    {
      return rcmsStateNotifier_.getRcmsStateListenerParameter();
    }

    // report if RCMS StateListener was found
    xdata::Boolean* foundRcmsStateListener()
    {
      return rcmsStateNotifier_.getFoundRcmsStateListenerParameter();
    }

    // initialize state machine and bind callbacks to driving application
    template<class T> void initialize(T *app)
    {
      // action signatures
      asConfiguring_ = toolbox::task::bind(app,&T::configuring,"configuring");
      asEnabling_    = toolbox::task::bind(app,&T::enabling,   "enabling");
      asStopping_    = toolbox::task::bind(app,&T::stopping,   "stopping");
      asHalting_     = toolbox::task::bind(app,&T::halting,    "halting");
      
      // work loops
      workLoopConfiguring_ =
	toolbox::task::getWorkLoopFactory()->getWorkLoop(appNameAndInstance_+
							 "_Configuring",
							 "waiting");
      workLoopEnabling_ =
	toolbox::task::getWorkLoopFactory()->getWorkLoop(appNameAndInstance_+
							 "_Enabling",
							 "waiting");
      workLoopStopping_ =
	toolbox::task::getWorkLoopFactory()->getWorkLoop(appNameAndInstance_+
							 "_Stopping",
							 "waiting");
      workLoopHalting_ =
	toolbox::task::getWorkLoopFactory()->getWorkLoop(appNameAndInstance_+
							 "_Halting",
							 "waiting");
      
      
      // bind SOAP callbacks
      xoap::bind(app,&T::fsmCallback,"Configure",XDAQ_NS_URI);
      xoap::bind(app,&T::fsmCallback,"Enable",   XDAQ_NS_URI);
      xoap::bind(app,&T::fsmCallback,"Stop",     XDAQ_NS_URI);
      xoap::bind(app,&T::fsmCallback,"Halt",     XDAQ_NS_URI);
      
      // define finite state machine, states&transitions
      fsm_.addState('h', "halting"    ,this,&evf::StateMachine::stateChanged);
      fsm_.addState('H', "Halted"     ,this,&evf::StateMachine::stateChanged);
      fsm_.addState('c', "configuring",this,&evf::StateMachine::stateChanged);
      fsm_.addState('R', "Ready"      ,this,&evf::StateMachine::stateChanged);
      fsm_.addState('e', "enabling"   ,this,&evf::StateMachine::stateChanged);
      fsm_.addState('E', "Enabled"    ,this,&evf::StateMachine::stateChanged);
      fsm_.addState('s', "stopping"   ,this,&evf::StateMachine::stateChanged);
      
      fsm_.addStateTransition('H','c',"Configure");
      fsm_.addStateTransition('c','R',"ConfigureDone");
      fsm_.addStateTransition('R','e',"Enable");
      fsm_.addStateTransition('e','E',"EnableDone");
      fsm_.addStateTransition('E','s',"Stop");
      fsm_.addStateTransition('s','R',"StopDone");
      fsm_.addStateTransition('E','h',"Halt");
      fsm_.addStateTransition('R','h',"Halt");
      fsm_.addStateTransition('h','H',"HaltDone");
      
      fsm_.addStateTransition('c','F',"Fail",this,&evf::StateMachine::failed);
      fsm_.addStateTransition('e','F',"Fail",this,&evf::StateMachine::failed);
      fsm_.addStateTransition('s','F',"Fail",this,&evf::StateMachine::failed);
      fsm_.addStateTransition('h','F',"Fail",this,&evf::StateMachine::failed);
      
      fsm_.addStateTransition('E','F',"Fail",this,&evf::StateMachine::failed);

      fsm_.setFailedStateTransitionAction(this,&evf::StateMachine::failed);
      fsm_.setFailedStateTransitionChanged(this,&evf::StateMachine::stateChanged);
      fsm_.setStateName('F',"Failed");
      
      fsm_.setInitialState('H');
      fsm_.reset();
      stateName_ = fsm_.getStateName(fsm_.getCurrentState());
      
      if (!workLoopConfiguring_->isActive()) workLoopConfiguring_->activate();
      if (!workLoopEnabling_->isActive())    workLoopEnabling_   ->activate();
      if (!workLoopStopping_->isActive())    workLoopStopping_   ->activate();
      if (!workLoopHalting_->isActive())     workLoopHalting_    ->activate();
    }
    
     
  private:
    //
    // member data
    //
    
    // application name&instance
    log4cplus::Logger                logger_;
    xdata::InfoSpace*                appInfoSpace_;
    std::string                      appNameAndInstance_;
    xdata::String                    stateName_;
    bool                             doStateNotification_;
    
    // finite state machine
    toolbox::fsm::FiniteStateMachine fsm_;
    
    // work loops for transitional states
    toolbox::task::WorkLoop         *workLoopConfiguring_;
    toolbox::task::WorkLoop         *workLoopEnabling_;
    toolbox::task::WorkLoop         *workLoopStopping_;
    toolbox::task::WorkLoop         *workLoopHalting_;

    // action signatures for transitional states
    toolbox::task::ActionSignature  *asConfiguring_;
    toolbox::task::ActionSignature  *asEnabling_;
    toolbox::task::ActionSignature  *asStopping_;
    toolbox::task::ActionSignature  *asHalting_;
    
    // rcms state notifier
    xdaq2rc::RcmsStateNotifier       rcmsStateNotifier_;
    
  };
  
}


#endif
