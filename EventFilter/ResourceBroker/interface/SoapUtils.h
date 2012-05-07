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
 * $Author: mommsen $
 * $Revision: 1.4 $
 * $Date: 2011/03/07 15:31:32 $
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
