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

#include "EventFilter/StorageManager/interface/SMStateMachine.h"
#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"

using namespace stor;
using namespace std;

SMStateMachine::SMStateMachine(log4cplus::Logger &logger) : logger_(logger)
{
}

xoap::MessageReference SMStateMachine::processFSMCommand(const string cmdName)
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
xoap::MessageReference SMStateMachine::createFSMReplyMsg(const string cmd, const string state)
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
