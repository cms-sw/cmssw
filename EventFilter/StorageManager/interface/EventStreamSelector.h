// -*- c++ -*-
// $Id$

#ifndef EVENTSTREAMSELECTOR_H
#define EVENTSTREAMSELECTOR_H

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor {

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
    EventStreamConfigurationInfo _configInfo;

    boost::shared_ptr<edm::EventSelector> _eventSelector;

  };

} // namespace stor

#endif
