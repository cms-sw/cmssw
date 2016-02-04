// $Id: SoapUtils.h,v 1.4 2011/03/07 15:31:32 mommsen Exp $
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
     * $Revision: 1.4 $
     * $Date: 2011/03/07 15:31:32 $
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
