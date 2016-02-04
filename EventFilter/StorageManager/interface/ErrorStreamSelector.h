// $Id: ErrorStreamSelector.h,v 1.4 2010/12/16 16:35:29 mommsen Exp $
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

     $Author: mommsen $
     $Revision: 1.4 $
     $Date: 2010/12/16 16:35:29 $
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

    // Comparison:
    bool operator<(const ErrorStreamSelector& other) const
    { return ( _configInfo < other.configInfo() ); }

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
