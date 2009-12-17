////////////////////////////////////////////////////////////////////////////////
//
// StateMachine
// ------------
//
//            03/06/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/Utilities/interface/StateMachine.h"
#include "EventFilter/Utilities/interface/FsmFailedEvent.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "toolbox/fsm/FailedEvent.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include "xcept/tools.h"

#include <typeinfo>
#include <string>
#include <sstream>


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
StateMachine::StateMachine(xdaq::Application* app)
  : logger_(app->getApplicationLogger())
  , appInfoSpace_(app->getApplicationInfoSpace())
  , doStateNotification_(true)
  , workLoopConfiguring_(0)
  , workLoopEnabling_(0)
  , workLoopStopping_(0)
  , workLoopHalting_(0)
  , asConfiguring_(0)
  , asEnabling_(0)
  , asStopping_(0)
  , asHalting_(0)
  , rcmsStateNotifier_(app->getApplicationLogger(),
		       app->getApplicationDescriptor(),
		       app->getApplicationContext())
{
  ostringstream oss;
  oss<<app->getApplicationDescriptor()->getClassName()
     <<app->getApplicationDescriptor()->getInstance();
  appNameAndInstance_ = oss.str();
}


//______________________________________________________________________________
StateMachine::~StateMachine()
{

}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
xoap::MessageReference StateMachine::commandCallback(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  xoap::SOAPPart     part    =msg->getSOAPPart();
  xoap::SOAPEnvelope env     =part.getEnvelope();
  xoap::SOAPBody     body    =env.getBody();
  DOMNode           *node    =body.getDOMNode();
  DOMNodeList       *bodyList=node->getChildNodes();
  DOMNode           *command =0;
  string             commandName;
  
  for (unsigned int i=0;i<bodyList->getLength();i++) {
    command = bodyList->item(i);
    if(command->getNodeType() == DOMNode::ELEMENT_NODE) {
      commandName = xoap::XMLCh2String(command->getLocalName());
      break;
    }
  }
  
  if (commandName.empty()) {
    XCEPT_RAISE(xoap::exception::Exception,"Command not found.");
  }
  
  // fire appropriate event and create according response message
  try {
    toolbox::Event::Reference e(new toolbox::Event(commandName,this));
    fsm_.fireEvent(e);
    
    // response string
    xoap::MessageReference reply = xoap::createMessage();
    xoap::SOAPEnvelope envelope  = reply->getSOAPPart().getEnvelope();
    xoap::SOAPName responseName  = envelope.createName(commandName+"Response",
						       "xdaq",XDAQ_NS_URI);
    xoap::SOAPBodyElement responseElem =
      envelope.getBody().addBodyElement(responseName);
    
    // state string
    int               iState        = fsm_.getCurrentState();
    string            state         = fsm_.getStateName(iState);
    xoap::SOAPName    stateName     = envelope.createName("state",
							  "xdaq",XDAQ_NS_URI);
    xoap::SOAPElement stateElem     = responseElem.addChildElement(stateName);
    xoap::SOAPName    attributeName = envelope.createName("stateName",
							  "xdaq",XDAQ_NS_URI);
    stateElem.addAttribute(attributeName,state);
    
    return reply;
  }
  catch (toolbox::fsm::exception::Exception & e) {
    XCEPT_RETHROW(xoap::exception::Exception,"invalid command.",e);
  }	
}


//______________________________________________________________________________
void StateMachine::stateChanged(toolbox::fsm::FiniteStateMachine & fsm) 
  throw (toolbox::fsm::exception::Exception)
{
  stateName_   = fsm_.getStateName(fsm_.getCurrentState());
  string state = stateName_.toString();
  
  LOG4CPLUS_INFO(logger_,"New state is: "<<state);
  
  if (state=="configuring") {
    try {
      workLoopConfiguring_->submit(asConfiguring_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(logger_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="enabling") {
    try {
      workLoopEnabling_->submit(asEnabling_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(logger_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="stopping") {
    try {
      workLoopStopping_->submit(asStopping_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(logger_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="halting") {
    try {
      workLoopHalting_->submit(asHalting_);
    }
    catch (xdaq::exception::Exception& e) {
      LOG4CPLUS_ERROR(logger_,xcept::stdformat_exception_history(e));
    }
  }
  else if (state=="Halted"||state=="Ready"||state=="Enabled"||state=="Failed") {
    if(doStateNotification_)
      {
	try {
	  rcmsStateNotifier_.stateChanged(state,appNameAndInstance_+
					  " has reached target state " +
					  state);
	}
	catch (xcept::Exception& e) {
	  LOG4CPLUS_ERROR(logger_,"Failed to notify state change: "
			  <<xcept::stdformat_exception_history(e));
	}
      }
  }
}


//______________________________________________________________________________
void StateMachine::failed(toolbox::Event::Reference e)
  throw (toolbox::fsm::exception::Exception)
{
  if (typeid(*e) == typeid(toolbox::fsm::FailedEvent)) {
    toolbox::fsm::FailedEvent &fe=dynamic_cast<toolbox::fsm::FailedEvent&>(*e);
    LOG4CPLUS_FATAL(logger_,"Failure occurred in transition from '"
		    <<fe.getFromState()<<"' to '"<<fe.getToState()
		    <<"', exception history: "
		    <<xcept::stdformat_exception_history(fe.getException()));
  }
  else if (typeid(*e) == typeid(evf::FsmFailedEvent)) {
    evf::FsmFailedEvent &fe=dynamic_cast<evf::FsmFailedEvent&>(*e);
    LOG4CPLUS_FATAL(logger_,"fsm failure occured: "<<fe.errorMessage());
  }
}


//______________________________________________________________________________
void StateMachine::fireEvent(const string& evtType,void* originator)
{
  toolbox::Event::Reference e(new toolbox::Event(evtType,originator));
  fsm_.fireEvent(e);
}


//______________________________________________________________________________
void StateMachine::fireFailed(const string& errorMsg,void* originator)
{
  toolbox::Event::Reference e(new evf::FsmFailedEvent(errorMsg,originator));
  fsm_.fireEvent(e);
}


//______________________________________________________________________________
void StateMachine::findRcmsStateListener()
{
  rcmsStateNotifier_.findRcmsStateListener(); //might not be needed
  rcmsStateNotifier_.subscribeToChangesInRcmsStateListener(appInfoSpace_); 
}
