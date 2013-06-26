/**
 * Collection of utility functions for handling SOAP messages
 *
 * $Author: aspataru $
 * $Revision: 1.3 $
 * $Date: 2012/05/03 09:37:32 $
 */

#include "EventFilter/ResourceBroker/interface/SoapUtils.h"

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

using std::string;

namespace evf {
namespace soaputils {

string extractParameters(xoap::MessageReference msg, xdaq::Application* app) {
	string command;

	// Extract the command name and update any configuration parameter
	// found in the SOAP message in the application infospace
	xdaq2rc::SOAPParameterExtractor soapParameterExtractor(app);
	command = soapParameterExtractor.extractParameters(msg);
	return command;
}

xoap::MessageReference createFsmSoapResponseMsg(const string commandName,
		const string currentState) {
	xoap::MessageReference reply;

	try {
		// response string
		reply = xoap::createMessage();
		xoap::SOAPEnvelope envelope = reply->getSOAPPart().getEnvelope();
		xoap::SOAPName responseName = envelope.createName(
				commandName + "Response", "xdaq", XDAQ_NS_URI);
		xoap::SOAPBodyElement responseElem = envelope.getBody().addBodyElement(
				responseName);

		// state string
		xoap::SOAPName stateName = envelope.createName("state", "xdaq",
				XDAQ_NS_URI);
		xoap::SOAPElement stateElem = responseElem.addChildElement(stateName);
		xoap::SOAPName attributeName = envelope.createName("stateName", "xdaq",
				XDAQ_NS_URI);
		stateElem.addAttribute(attributeName, currentState);
	} catch (xcept::Exception &e) {
		XCEPT_RETHROW(
				xoap::exception::Exception,
				"Failed to create FSM SOAP response message for command '"
						+ commandName + "' and current state '" + currentState
						+ "'.", e);
	}

	return reply;
}

} // namespace soaputils

} // namespace evf
