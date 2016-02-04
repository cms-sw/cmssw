#include "DQMServices/XdaqCollector/interface/StateMachine.h" 
#include "log4cplus/logger.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"

#include <string>

using namespace dqm;

StateMachine::StateMachine(xdaq::ApplicationStub *s) :  
  xdaq::Application(s), 
  fsm_(getApplicationLogger()) 
{      
  wsm_.addState('H', "Halted"   , this, &StateMachine::statePage);
  wsm_.addState('R', "Ready"    , this, &StateMachine::statePage);
  wsm_.addState('E', "Enabled"  , this, &StateMachine::statePage);
  wsm_.addState('S', "Suspended", this, &StateMachine::statePage);
  
  // Define FSM transitions
  wsm_.addStateTransition('H', 'R', "Configure", this,
			  &StateMachine::webConfigure, &StateMachine::failurePage);
  wsm_.addStateTransition('R', 'E', "Enable", this,
			  &StateMachine::webEnable, &StateMachine::failurePage);
  wsm_.addStateTransition('E', 'S', "Suspend", this,
			  &StateMachine::webSuspend, &StateMachine::failurePage);
  wsm_.addStateTransition('S', 'E', "Resume", this,
			  &StateMachine::webResume, &StateMachine::failurePage);
  wsm_.addStateTransition('H', 'H', "Halt", this,
			  &StateMachine::webHalt, &StateMachine::failurePage);
  wsm_.addStateTransition('R', 'H', "Halt", this,
			  &StateMachine::webHalt, &StateMachine::failurePage);
  wsm_.addStateTransition('E', 'H', "Halt", this,
			  &StateMachine::webHalt, &StateMachine::failurePage);
  wsm_.addStateTransition('S', 'H', "Halt", this,
			  &StateMachine::webHalt, &StateMachine::failurePage);
  
  wsm_.setInitialState('H');
  
  fsm_.init<dqm::StateMachine>(this);
  xgi::bind(this, &StateMachine::dispatch, "dispatch");
}

void StateMachine::bind(std::string page)
{
  page_ = page;
  xgi::bind(this, &StateMachine::Default, page);
}   

xoap::MessageReference StateMachine::fireEvent(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  xoap::SOAPPart     part      = msg->getSOAPPart();
  xoap::SOAPEnvelope env       = part.getEnvelope();
  xoap::SOAPBody     body      = env.getBody();
  DOMNode            *node     = body.getDOMNode();
  DOMNodeList        *bodyList = node->getChildNodes();
  DOMNode            *command  = 0;
  std::string        commandName;
  
  for (unsigned int i = 0; i < bodyList->getLength(); i++)
    {
      command = bodyList->item(i);
      
      if(command->getNodeType() == DOMNode::ELEMENT_NODE)
	{
	  commandName = xoap::XMLCh2String(command->getLocalName());
	  wsm_.fireEvent(commandName,0,0);
	  return fsm_.processFSMCommand(commandName);
	}
    }
  
  XCEPT_RAISE(xoap::exception::Exception, "Command not found");
}

//
// Web Navigation Pages
//
void StateMachine::statePage( xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  if(out)
    {
      std::string url = "/";
      url += getApplicationDescriptor()->getURN();
      std::string purl = url + "/" + page_;

      *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
      *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
      *out << "<head>" << std::endl;
      *out << "<META HTTP-EQUIV=refresh CONTENT=\"30; URL=";
      *out << purl << "\">" << std::endl;
      *out << "<META HTTP-EQUIV=Window-target CONTENT=\"_self\">" << std::endl;
      *out << "<title> " << "fsm" << "</title>"  << std::endl;
      *out << "</head>" << std::endl;
      *out << "<body>" << std::endl;
      //      xgi::Utils::getPageHeader(*out, "StateMachine", wsm_.getStateName(wsm_.getCurrentState()));
      
      url += "/dispatch";	
      
      // display FSM
      //      cout << "current state " << wsm_.getCurrentState() << endl;
      //      cout << "possible commands" << endl;
      std::set<std::string>::iterator i;
      std::set<std::string> possibleInputs = wsm_.getInputs(wsm_.getCurrentState());
      std::set<std::string> allInputs = wsm_.getInputs();
      
      //      for ( i = possibleInputs.begin(); i != possibleInputs.end(); i++)
      //	cout << (*i) << endl;
      
      *out << cgicc::h3("Finite State Machine").set("style", "font-family: arial") << std::endl;
      *out << "<table border cellpadding=10 cellspacing=0>" << std::endl;
      *out << "<tr>" << std::endl;
      *out << "<th>" << wsm_.getStateName(wsm_.getCurrentState()) << "</th>" << std::endl;
      *out << "</tr>" << std::endl;
      *out << "<tr>" << std::endl;

      for ( i = allInputs.begin(); i != allInputs.end(); i++)
	{
	  *out << "<td>"; 
	  *out << cgicc::form().set("method","post").set("target","_self").set("action", url).set("enctype","multipart/form-data") << std::endl;
	  
	  if ( possibleInputs.find(*i) != possibleInputs.end() )
	    {
	      *out << cgicc::input().set("type", "submit").set("name", "StateInput").set("value", (*i) );
	    }
	  else
	    {
	      *out << cgicc::input() .set("type", "submit").set("name", "StateInput").set("value", (*i) ).set("disabled", "true");
	    }
	  
	  *out << cgicc::form();
	  *out << "</td>" << std::endl;
	}
      
      *out << "</tr>" << std::endl;
      *out << "</table>" << std::endl;
      *out << cgicc::html();
      //	
      //      xgi::Utils::getPageFooter(*out);
    }
}

//
// Failure Pages
//
void StateMachine::failurePage(xgi::Output * out, xgi::exception::Exception & e)  
  throw (xgi::exception::Exception)
{
  if(out)
    {
      *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
      *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
      
      xgi::Utils::getPageHeader(*out, "WebStateMachine Home", "Failure");
      
      
      *out << cgicc::br() << e.what() << cgicc::br() << std::endl;
      std::string url = "/";
      url += getApplicationDescriptor()->getURN();

      *out << cgicc::br() << "<a href=\"" << url << "\">" << "retry" << "</a>" << cgicc::br() << std::endl;
      
      xgi::Utils::getPageFooter(*out);
      
    }
}
  
