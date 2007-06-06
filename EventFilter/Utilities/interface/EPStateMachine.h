#ifndef EPSTATEMACHINE_H
#define EPSTATEMACHINE_H

#include "toolbox/fsm/FiniteStateMachine.h"
#include "toolbox/fsm/FailedEvent.h"
#include "xcept/tools.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"
#include "log4cplus/logger.h"
#include "xdata/String.h"
#include "xdaq/NamespaceURI.h"

namespace evf
{
  class EPStateMachine : public toolbox::fsm::FiniteStateMachine
    {
    public:
      EPStateMachine(log4cplus::Logger &logger);
      /**
       * Application state as an "integer" - to be used in the application
       */
      toolbox::fsm::State state_;


      /**
       * Application state as a string - to be used an exported parameter for
       * run-control.
       */
      xdata::String stateName_;


      template<class T> void init(T*me)
	{
	  // Define FSM states
	  addState('H', "Halted"   , this, &EPStateMachine::stateChanged);
	  addState('R', "Ready"    , this, &EPStateMachine::stateChanged);
	  addState('E', "Enabled"  , this, &EPStateMachine::stateChanged);
	  addState('S', "Suspended", this, &EPStateMachine::stateChanged);
	  
	  // Define FSM transitions
	  addStateTransition('H', 'R', "Configure", me, &T::configureAction);
	  addStateTransition('R', 'E', "Enable",    me, &T::enableAction);
	  addStateTransition('E', 'R', "Stop",      me, &T::stopAction);
	  addStateTransition('E', 'S', "Suspend",   me, &T::suspendAction);
	  addStateTransition('S', 'E', "Resume",    me, &T::resumeAction);
	  addStateTransition('H', 'H', "Halt",      me, &T::nullAction);
	  addStateTransition('R', 'H', "Halt",      me, &T::haltAction);
	  addStateTransition('E', 'H', "Halt",      me, &T::haltAction);
	  addStateTransition('S', 'H', "Halt",      me, &T::haltAction);
	  
	  setFailedStateTransitionAction(this,&EPStateMachine::failedTransition);
	  setFailedStateTransitionChanged(this,&EPStateMachine::stateChanged);
	  
	  setInitialState('H');
	  reset();
	  
	  xoap::bind(me,&T::fireEvent,"Configure", XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Stop"     , XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Enable"   , XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Suspend"  , XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Resume"   , XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Halt"     , XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Disable"  , XDAQ_NS_URI); 
	  xoap::bind(me,&T::fireEvent,"Fail"     , XDAQ_NS_URI); 
	}


      void failedTransition(toolbox::Event::Reference e)
	throw (toolbox::fsm::exception::Exception)
      {
	toolbox::fsm::FailedEvent &fe =
	  dynamic_cast<toolbox::fsm::FailedEvent&>(*e);
	
	LOG4CPLUS_FATAL(logger_,
			"Failure occurred when performing transition from: "
			<< fe.getFromState() <<  " to: " << fe.getToState()
			<< " exception: " << fe.getException().what());
      }
      
      void stateChanged(toolbox::fsm::FiniteStateMachine & fsm)
	throw (toolbox::fsm::exception::Exception)
      {
	LOG4CPLUS_INFO(logger_,
		       "Changed to state: "
		       << getStateName(getCurrentState()));
      }
      
      
      /**
       * Calls FiniteStateMachine::reset() and keeps stateName_ and state_
       * in sync.
       */
      void reset() throw (toolbox::fsm::exception::Exception)
      {
	FiniteStateMachine::reset();
	
	state_     = FiniteStateMachine::getCurrentState();
	stateName_ = FiniteStateMachine::getStateName(state_);
      }
      
      
      /**
       * Calls FiniteStateMachine::fireEvent() and keeps stateName_ and state_
       * in sync.
       */
      void fireEvent(toolbox::Event::Reference e) 
	throw (toolbox::fsm::exception::Exception)
      {
	try{
	  FiniteStateMachine::fireEvent(e);
	}
	catch(toolbox::fsm::exception::Exception ex)
	  {
	    LOG4CPLUS_ERROR(logger_,"EPStateMachine fireEvent failed " 
			    << ex.what());
	    }  
	catch(...)
	  {
	      LOG4CPLUS_ERROR(logger_,"EPStateMachine fireEvent failed " 
			      << " Unknown Exception " << " state is " 
			      << FiniteStateMachine::getCurrentState());
	  }
	
	
	state_     = FiniteStateMachine::getCurrentState();
	stateName_ = FiniteStateMachine::getStateName(state_);
      }
      
      xoap::MessageReference processFSMCommand(const std::string cmdName)
	throw (xoap::exception::Exception);
      xoap::MessageReference createFSMReplyMsg(const std::string cmd, 
					       const std::string state);

    private:
      log4cplus::Logger &logger_;

    };
}
#endif
