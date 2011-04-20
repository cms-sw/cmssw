// $Id: ErrorStreamSelector.h,v 1.4.2.2 2011/02/28 17:56:15 mommsen Exp $
/// @file: ErrorStreamSelector.h 

#ifndef EventFilter_StorageManager_ErrorStreamSelector_h
#define EventFilter_StorageManager_ErrorStreamSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

namespace stor {

  /**
     Accepts or rejects an error event based on the 
     ErrorStreamConfigurationInfo

     $Author: mommsen $
     $Revision: 1.4.2.2 $
     $Date: 2011/02/28 17:56:15 $
  */

  class ErrorStreamSelector
  {

  public:

    // Constructor:
    ErrorStreamSelector( const ErrorStreamConfigurationInfo& configInfo ):
      configInfo_( configInfo )
    {}

    // Destructor:
    ~ErrorStreamSelector() {}

    // Accept event:
    bool acceptEvent( const I2OChain& );

    // Accessors:
    const ErrorStreamConfigurationInfo& configInfo() const { return configInfo_; }

    // Comparison:
    bool operator<(const ErrorStreamSelector& other) const
    { return ( configInfo_ < other.configInfo() ); }

  private:

    ErrorStreamConfigurationInfo configInfo_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_ErrorStreamSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
