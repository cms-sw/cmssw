// $Id: SoapUtils.h,v 1.3.18.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: SoapUtils.h 

#ifndef EventFilter_StorageManager_SoapUtils_h
#define EventFilter_StorageManager_SoapUtils_h

#include "xdaq/Application.h"
#include "xoap/MessageReference.h"

#include <string>

namespace stor {

  namespace soaputils {

    /**
     * Collection of utility functions for handling SOAP messages
     *
     * $Author: mommsen $
     * $Revision: 1.3.18.1 $
     * $Date: 2011/03/07 11:33:04 $
     */

    /**
     * Extract parameters and FSM command from SOAP message
     */
    std::string extractParameters
    (
      xoap::MessageReference,
      xdaq::Application*
    );

    /**
     * Create a SOAP FSM response message
     */
    xoap::MessageReference createFsmSoapResponseMsg
    (
      const std::string commandName,
      const std::string currentState
    );

  } // namespace soaputils
  
} // namespace stor

#endif // EventFilter_StorageManager_SoapUtils_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
