// $Id: EventDistributor.h,v 1.8 2011/03/07 15:31:31 mommsen Exp $
/// @file: EventDistributor.h 

#ifndef EventFilter_StorageManager_EventDistributor_h
#define EventFilter_StorageManager_EventDistributor_h

#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"

#include "boost/shared_ptr.hpp"


namespace stor {

  class DataSenderMonitorCollection;
  class DQMEventSelector;
  class ErrorStreamConfigurationInfo;
  class ErrorStreamSelector;
  class EventConsumerSelector;
  class EventStreamConfigurationInfo;
  class EventStreamSelector;
  class I2OChain;
  class QueueID;
  class StatisticsReporter;


  /**
   * Distributes complete events to appropriate queues
   *
   * It receives complete events in form of I2OChains and
   * distributes it to the appropriate queues by checking
   * the I2O message type and the trigger bits in the event
   * header.
   *
   * $Author: mommsen $
   * $Revision: 1.8 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class EventDistributor
  {
  public:

    EventDistributor(SharedResourcesPtr sr);

    ~EventDistributor();

    /**
     * Add the event given as I2OChain to the appropriate queues
     */
    void addEventToRelevantQueues( I2OChain& );

    /**
     * Returns true if no further events can be processed,
     * e.g. the StreamQueue is full
     */
    const bool full() const;

    /**
     * Registers a new consumer
     */
    void registerEventConsumer( const EventConsRegPtr );

    /**
     * Registers a new DQM consumer
     */
    void registerDQMEventConsumer( const DQMEventConsRegPtr );

    /**
     * Registers the full set of event streams.
     */
    void registerEventStreams( const EvtStrConfigListPtr );

    /**
     * Registers the full set of error event streams.
     */
    void registerErrorStreams( const ErrStrConfigListPtr );

    /**
     * Clears out all existing event and error streams.
     */
    void clearStreams();

    /**
     * Returns the number of streams that have been configured.
     */
    unsigned int configuredStreamCount() const;

    /**
     * Returns the number of streams that have been configured and initialized.
     */
    unsigned int initializedStreamCount() const;

    /**
     * Clears out all existing consumer registrations.
     */
    void clearConsumers();

    /**
     * Returns the number of consumers that have been configured.
     */
    unsigned int configuredConsumerCount() const;

    /**
     * Returns the number of consumers that have been configured and initialized.
     */
    unsigned int initializedConsumerCount() const;

    /**
       Updates staleness info for consumers.
    */
    void checkForStaleConsumers();

  private:

    void tagCompleteEventForQueues( I2OChain& );

    SharedResourcesPtr sharedResources_;

    typedef boost::shared_ptr<EventStreamSelector> EvtSelPtr;
    typedef std::set<EvtSelPtr, utils::ptrComp<EventStreamSelector> > EvtSelList;
    EvtSelList eventStreamSelectors_;

    typedef boost::shared_ptr<DQMEventSelector> DQMEvtSelPtr;
    typedef std::set<DQMEvtSelPtr, utils::ptrComp<DQMEventSelector> > DQMEvtSelList;
    DQMEvtSelList dqmEventSelectors_;

    typedef boost::shared_ptr<ErrorStreamSelector> ErrSelPtr;
    typedef std::set<ErrSelPtr, utils::ptrComp<ErrorStreamSelector> > ErrSelList;
    ErrSelList errorStreamSelectors_;

    typedef boost::shared_ptr<EventConsumerSelector> ConsSelPtr;
    typedef std::set<ConsSelPtr, utils::ptrComp<EventConsumerSelector> > ConsSelList;
    ConsSelList eventConsumerSelectors_;

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_EventDistributor_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
