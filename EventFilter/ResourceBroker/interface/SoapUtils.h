#ifndef EVF_RB_SOAPUTILS_H
#define EVF_RB_SOAPUTILS_H

#include "xdaq/Application.h"
#include "xoap/MessageReference.h"

#include <string>

namespace evf {

namespace soaputils {

/**
 * Collection of utility functions for handling SOAP messages
 *
 * $Author: aspataru $
 * $Revision: 1.1.2.10 $
 * $Date: 2012/04/21 00:23:00 $
 */

/**
 * Extract parameters and FSM command from SOAP message
 */
std::string extractParameters(xoap::MessageReference, xdaq::Application*);

/**
 * Create a SOAP FSM response message
 */
xoap::MessageReference createFsmSoapResponseMsg(const std::string commandName,
		const std::string currentState);

} // namespace soaputils

} // namespace evf

#endif // EVF_RB_SOAPUTILS_H
