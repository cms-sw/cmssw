// $Id: EventConsumerSelector.h,v 1.9 2011/03/07 15:31:31 mommsen Exp $
/// @file: EventConsumerSelector.h 

#ifndef EventFilter_StorageManager_EventConsumerSelector_h
#define EventFilter_StorageManager_EventConsumerSelector_h

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
   * $Revision: 1.9 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class EventConsumerSelector
  {

  public:

    /**
     * Constructs an EventConsumerSelector instance based on the
     * specified registration information.
     */
    EventConsumerSelector( const EventConsRegPtr registrationInfo ):
      initialized_( false ),
      outputModuleId_( 0 ),
      registrationInfo_( registrationInfo ),
      acceptedEvents_( 0 )
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
    QueueID const queueId() const { return registrationInfo_->queueId(); }

    /**
     * Tests whether this selector has been initialized.
     */
    bool isInitialized() const { return initialized_; }

    /**
     *  Comparison:
     */
    bool operator<(const EventConsumerSelector& other) const;

  private:

    bool initialized_;
    unsigned int outputModuleId_;
    const EventConsRegPtr registrationInfo_;
    TriggerSelectorPtr eventSelector_;
    unsigned long acceptedEvents_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_EventConsumerSelector_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
