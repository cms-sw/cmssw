// $Id: EventStreamSelector.h,v 1.6 2009/12/01 13:58:08 mommsen Exp $
/// @file: EventStreamSelector.h 

#ifndef StorageManager_EventStreamSelector_h
#define StorageManager_EventStreamSelector_h

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
     $Revision: 1.6 $
     $Date: 2009/12/01 13:58:08 $
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
    unsigned int outputModuleId() const { return _outputModuleId; }
    const EventStreamConfigurationInfo& configInfo() const { return _configInfo; }
    bool isInitialized() const { return _initialized; }

    // Comparison:
    bool operator<(const EventStreamSelector& other) const
    { return ( _configInfo < other.configInfo() ); }

  private:

    bool _initialized;
    unsigned int _outputModuleId;
    const EventStreamConfigurationInfo _configInfo;

    boost::shared_ptr<TriggerSelector> _eventSelector;

  };

} // namespace stor

#endif // StorageManager_EventStreamSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
