// $Id: Utils.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $

#ifndef StorageManager_SoapUtils_h
#define StorageManager_SoapUtils_h

#include "xoap/MessageReference.h"

#include <string>

namespace stor {

  namespace soaputils {

    /**
     * Collection of utility functions for handling SOAP messages
     *
     * $Author: dshpakov $
     * $Revision: 1.2 $
     * $Date: 2009/06/10 08:15:24 $
     */

    /**
     * Extract parameters and FSM command from SOAP message
     */
    std::string extractParameters( xoap::MessageReference );

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

#endif // StorageManager_SoapUtils_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
