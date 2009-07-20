// $Id: EventConsumerSelector.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $
/// @file: EventConsumerSelector.h 

#ifndef StorageManager_EventConsumerSelector_h
#define StorageManager_EventConsumerSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor {

  /**
   * Defines the common interface for event and DQM consumer
   * registration info objects.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:22 $
   */

  class EventConsumerSelector
  {

  public:

    /**
     * Constructs an EventConsumerSelector instance based on the
     * specified registration information.
     */
    EventConsumerSelector( const EventConsumerRegistrationInfo* configInfo ):
      _initialized( false ),
      _stale( false ),
      _outputModuleId( 0 ),
      _outputModuleLabel( configInfo->selHLTOut() ),
      _eventSelectionStrings( configInfo->selEvents() ),
      _queueId( configInfo->queueId() )
    {}

    /**
     * Destructs the EventConsumerSelector instance.
     */
    ~EventConsumerSelector() {}

    /**
     * Initializes the selector instance from the specified
     * INIT message.  EventConsumerSelector instances need to be
     * initialized before they will accept any events.
     */
    void initialize( const InitMsgView& );

    /**
     * Tests whether the specified event is accepted by this selector -
     * returns true if the event is accepted, false otherwise.
     */
    bool acceptEvent( const I2OChain& );

    /**
     * Returns the ID of the queue corresponding to this selector.
     */
    QueueID const& queueId() const { return _queueId; }

    /**
     * Tests whether this selector has been initialized.
     */
    bool isInitialized() const { return _initialized; }

    /**
       Check if stale:
    */
    bool isStale() const { return _stale; }

    /**
       Mark as stale:
    */
    void markAsStale() { _stale = true; }

    /**
       Mark as active:
    */
    void markAsActive() { _stale = false; }

  private:

    bool _initialized;
    bool _stale;
    unsigned int _outputModuleId;
    std::string _outputModuleLabel;
    Strings _eventSelectionStrings;
    QueueID _queueId;

    boost::shared_ptr<edm::EventSelector> _eventSelector;

  };

} // namespace stor

#endif // StorageManager_EventConsumerSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
