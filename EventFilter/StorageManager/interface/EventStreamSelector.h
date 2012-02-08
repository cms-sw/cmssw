// $Id: EventStreamSelector.h,v 1.7.4.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: EventStreamSelector.h 

#ifndef EventFilter_StorageManager_EventStreamSelector_h
#define EventFilter_StorageManager_EventStreamSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/TriggerSelector.h"
#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor {

  /**
     Accepts or rejects an event based on the 
     EventStreamConfigurationInfo

     $Author: mommsen $
     $Revision: 1.7.4.1 $
     $Date: 2011/03/07 11:33:04 $
  */

  class EventStreamSelector
  {

  public:

    // Constructor:
    EventStreamSelector( const EventStreamConfigurationInfo& );

    // Destructor:
    ~EventStreamSelector() {}

    // Initialize:
    void initialize( const InitMsgView& );

    // Accept event:
    bool acceptEvent( const I2OChain& );

    // Accessors:
    unsigned int outputModuleId() const { return outputModuleId_; }
    const EventStreamConfigurationInfo& configInfo() const { return configInfo_; }
    bool isInitialized() const { return initialized_; }

    // Comparison:
    bool operator<(const EventStreamSelector& other) const
    { return ( configInfo_ < other.configInfo() ); }

  private:

    bool initialized_;
    unsigned int outputModuleId_;
    const EventStreamConfigurationInfo configInfo_;

    TriggerSelectorPtr eventSelector_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_EventStreamSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
