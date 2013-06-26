/**
 * $Id: SoapUtils.cc,v 1.4 2010/05/03 13:51:09 mommsen Exp $
/// @file: SoapUtils.cc
 */

#include "EventFilter/StorageManager/interface/SoapUtils.h"

#include "xdaq/NamespaceURI.h"
#include "xoap/MessageFactory.h"
#include "xoap/Method.h"
#include "xoap/SOAPBody.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPName.h"
#include "xoap/SOAPPart.h"
#include "xoap/domutils.h"

#include "xdaq2rc/version.h"
#include "xdaq2rc/SOAPParameterExtractor.hh"

namespace stor
{
  namespace soaputils
  {

    std::string extractParameters( xoap::MessageReference msg, xdaq::Application* app )
    {
      std::string command;

      // Extract the command name and update any configuration parameter
      // found in the SOAP message in the application infospace
      xdaq2rc::SOAPParameterExtractor soapParameterExtractor(app);
      command = soapParameterExtractor.extractParameters(msg);
      return command;
    }


    xoap::MessageReference createFsmSoapResponseMsg
    (
      const std::string commandName,
      const std::string currentState
    )
    {
      xoap::MessageReference reply;
      
      try
      {
        // response string
        reply = xoap::createMessage();
        xoap::SOAPEnvelope envelope  = reply->getSOAPPart().getEnvelope();
        xoap::SOAPName responseName  = envelope.createName(commandName+"Response",
          "xdaq",XDAQ_NS_URI);
        xoap::SOAPBodyElement responseElem =
          envelope.getBody().addBodyElement(responseName);
        
        // state string
        xoap::SOAPName stateName = envelope.createName("state", "xdaq",XDAQ_NS_URI);
        xoap::SOAPElement stateElem = responseElem.addChildElement(stateName);
        xoap::SOAPName attributeName = envelope.createName("stateName", "xdaq",XDAQ_NS_URI);
        stateElem.addAttribute(attributeName,currentState);
      }
      catch(xcept::Exception &e)
      {
        XCEPT_RETHROW(xoap::exception::Exception,
          "Failed to create FSM SOAP response message for command '" +
          commandName + "' and current state '" + currentState + "'.",  e);
      }
      
      return reply;
    }
    
  } // namespace soaputils

} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
