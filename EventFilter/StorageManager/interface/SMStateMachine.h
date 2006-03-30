#ifndef SMSTATEMACHINE_H
#define SMSTATEMACHINE_H

/*
  This is the finite state machine code for the Storage Manager,
  copied from EventFilter/Utilities/interface/EPStateMachine.h
  and modified for Storage Manager states.
    Valid states are:
  Halted  - SM is stopped and disabled
  Ready   - SM is configured with the right parameterSet and ready to run
  Enabled - SM is running and receiving data via I2O, Event server can serve events

  HWKC - 27 Mar 2006
*/

#include "toolbox/include/toolbox/fsm/FiniteStateMachine.h"
#include "toolbox/include/toolbox/fsm/FailedEvent.h"
#include "xcept/include/xcept/tools.h"
#include "xoap/include/xoap/MessageFactory.h"
#include "xoap/include/xoap/Method.h"
#include "log4cplus/logger.h"
#include "xdata/include/xdata/String.h"
#include "xdaq/include/xdaq/NamespaceURI.h"

namespace stor
{
  class SMStateMachine : public toolbox::fsm::FiniteStateMachine
    {
    public:
      SMStateMachine(log4cplus::Logger &logger);
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
	  addState('H', "Halted"   , this, &SMStateMachine::stateChanged);
	  addState('R', "Ready"    , this, &SMStateMachine::stateChanged);
	  addState('E', "Enabled"  , this, &SMStateMachine::stateChanged);
	  
	  // Define FSM transitions
	  addStateTransition('H', 'R', "Configure", me,
			     &T::configureAction);
	  addStateTransition('R', 'E', "Enable", me,
			     &T::enableAction);
	  addStateTransition('H', 'H', "Halt", me,
			     &T::nullAction);
	  addStateTransition('R', 'H', "Halt", me,
			     &T::haltAction);
	  addStateTransition('E', 'H', "Halt", me,
			     &T::haltAction);
	  
	  setFailedStateTransitionAction
	    (
	     this,
	     &SMStateMachine::failedTransition
	     );
	  
	  setFailedStateTransitionChanged
	    (
	     this,
	     &SMStateMachine::stateChanged
	     );
	  
	  setInitialState('H');
	  reset();
	  xoap::bind(me,&T::fireEvent,"Configure", XDAQ_NS_URI);
	  xoap::bind(me,&T::fireEvent,"Enable"   , XDAQ_NS_URI);
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
	      LOG4CPLUS_ERROR(logger_,"SMStateMachine fireEvent failed " 
			      << ex.what());
	    }  
	  catch(...)
	    {
	      LOG4CPLUS_ERROR(logger_,"SMStateMachine fireEvent failed " 
			      << " Unknown Exception " << " state is " 
			      << FiniteStateMachine::getCurrentState());
	    }
	      
	  
	  state_     = FiniteStateMachine::getCurrentState();
	  stateName_ = FiniteStateMachine::getStateName(state_);
	}

      xoap::MessageReference processFSMCommand(const std::string cmdName)  throw (xoap::exception::Exception) ;
      xoap::MessageReference createFSMReplyMsg(const std::string cmd, 
					       const std::string state);

    private:
      log4cplus::Logger &logger_;

    };
}
#endif
