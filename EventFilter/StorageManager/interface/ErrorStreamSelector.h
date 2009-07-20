// $Id: ErrorStreamSelector.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $
/// @file: ErrorStreamSelector.h 

#ifndef StorageManager_ErrorStreamSelector_h
#define StorageManager_ErrorStreamSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
     Accepts or rejects an error event based on the 
     ErrorStreamConfigurationInfo

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */

  class ErrorStreamSelector
  {

  public:

    // Constructor:
    ErrorStreamSelector( const ErrorStreamConfigurationInfo& configInfo ):
      _configInfo( configInfo )
    {}

    // Destructor:
    ~ErrorStreamSelector() {}

    // Accept event:
    bool acceptEvent( const I2OChain& );

    // Accessors:
    const ErrorStreamConfigurationInfo& configInfo() const { return _configInfo; }

  private:

    ErrorStreamConfigurationInfo _configInfo;

  };

} // namespace stor

#endif // StorageManager_ErrorStreamSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
