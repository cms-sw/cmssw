#include "EventFilter/Utilities/interface/EPStateMachine.h"
#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"

using namespace evf;
using namespace std;

EPStateMachine::EPStateMachine(log4cplus::Logger &logger) : logger_(logger)
{
}

xoap::MessageReference EPStateMachine::processFSMCommand(const string cmdName)
  throw (xoap::exception::Exception)
{
  try
    {
      // Change state, calling the appropriate action method
      toolbox::Event::Reference evtRef(new toolbox::Event(cmdName, this));
      fireEvent(evtRef);
      return createFSMReplyMsg(cmdName, stateName_);
    }
  catch(xcept::Exception e)
    {
      XCEPT_RETHROW(xoap::exception::Exception,
		    "Failed to process " + cmdName, e);
    }
  catch(...)
    {
      XCEPT_RAISE(xoap::exception::Exception,
		  "Failed to process " + cmdName + " - Unknown exception");
    }
}
xoap::MessageReference EPStateMachine::createFSMReplyMsg(const string cmd, const string state)
{
  xoap::MessageReference msg = xoap::createMessage();
  xoap::SOAPEnvelope     env   = msg->getSOAPPart().getEnvelope();
  xoap::SOAPBody         body  = env.getBody();
  string                 rStr  = cmd + "Response";
  xoap::SOAPName         rName = env.createName(rStr,"xdaq",XDAQ_NS_URI);
  xoap::SOAPBodyElement  rElem = body.addBodyElement(rName);
  xoap::SOAPName       sName = env.createName("state","xdaq",XDAQ_NS_URI);
  xoap::SOAPElement      sElem = rElem.addChildElement(sName);
  xoap::SOAPName   aName = env.createName("stateName","xdaq",XDAQ_NS_URI);
  
  
  sElem.addAttribute(aName, state);
  
  return msg;
}
