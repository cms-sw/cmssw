// $Id: EventConsumerSelector.h,v 1.7 2010/12/17 18:21:04 mommsen Exp $
/// @file: EventConsumerSelector.h 

#ifndef StorageManager_EventConsumerSelector_h
#define StorageManager_EventConsumerSelector_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/TriggerSelector.h"
#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor {

  /**
   * Defines the common interface for event and DQM consumer
   * registration info objects.
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2010/12/17 18:21:04 $
   */

  class EventConsumerSelector
  {

  public:

    /**
     * Constructs an EventConsumerSelector instance based on the
     * specified registration information.
     */
    EventConsumerSelector( const EventConsumerRegistrationInfo* registrationInfo ):
      _initialized( false ),
      _stale( false ),
      _outputModuleId( 0 ),
      _registrationInfo( *registrationInfo ),
      _acceptedEvents( 0 )
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
    QueueID const queueId() const { return _registrationInfo.queueId(); }

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

    /**
       Comparison:
    */
    bool operator<(const EventConsumerSelector& other) const;

  private:

    bool _initialized;
    bool _stale;
    unsigned int _outputModuleId;
    const EventConsumerRegistrationInfo _registrationInfo;
    boost::shared_ptr<TriggerSelector> _eventSelector;
    unsigned long _acceptedEvents;

  };

} // namespace stor

#endif // StorageManager_EventConsumerSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
