// $Id: EventStreamSelector.h,v 1.3 2009/07/20 13:06:10 mommsen Exp $
/// @file: EventStreamSelector.h 

#ifndef StorageManager_EventStreamSelector_h
#define StorageManager_EventStreamSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor {

  /**
     Accepts or rejects an event based on the 
     EventStreamConfigurationInfo

     $Author: mommsen $
     $Revision: 1.3 $
     $Date: 2009/07/20 13:06:10 $
  */

  class EventStreamSelector
  {

  public:

    // Constructor:
    EventStreamSelector( const EventStreamConfigurationInfo& configInfo ):
      _initialized( false ),
      _outputModuleId(0),
      _configInfo( configInfo )
    {}

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

  private:

    bool _initialized;
    unsigned int _outputModuleId;
    const EventStreamConfigurationInfo _configInfo;

    boost::shared_ptr<edm::EventSelector> _eventSelector;

  };

} // namespace stor

#endif // StorageManager_EventStreamSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
