// $Id: EventDistributor.h,v 1.6 2010/03/03 15:19:17 mommsen Exp $
/// @file: EventDistributor.h 

#ifndef StorageManager_EventDistributor_h
#define StorageManager_EventDistributor_h

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"

#include "boost/shared_ptr.hpp"


namespace stor {

  class DataSenderMonitorCollection;
  class DQMEventConsumerRegistrationInfo;
  class DQMEventSelector;
  class ErrorStreamConfigurationInfo;
  class ErrorStreamSelector;
  class EventConsumerRegistrationInfo;
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
   * $Revision: 1.6 $
   * $Date: 2010/03/03 15:19:17 $
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
    void registerEventConsumer( const EventConsumerRegistrationInfo* );

    /**
     * Registers a new DQM consumer
     */
    void registerDQMEventConsumer( const DQMEventConsumerRegistrationInfo* );

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

    SharedResourcesPtr _sharedResources;

    typedef boost::shared_ptr<EventStreamSelector> EvtSelPtr;
    typedef std::set<EvtSelPtr, utils::ptr_comp<EventStreamSelector> > EvtSelList;
    EvtSelList _eventStreamSelectors;

    typedef boost::shared_ptr<DQMEventSelector> DQMEvtSelPtr;
    typedef std::set<DQMEvtSelPtr, utils::ptr_comp<DQMEventSelector> > DQMEvtSelList;
    DQMEvtSelList _dqmEventSelectors;

    typedef boost::shared_ptr<ErrorStreamSelector> ErrSelPtr;
    typedef std::set<ErrSelPtr, utils::ptr_comp<ErrorStreamSelector> > ErrSelList;
    ErrSelList _errorStreamSelectors;

    typedef boost::shared_ptr<EventConsumerSelector> ConsSelPtr;
    typedef std::set<ConsSelPtr, utils::ptr_comp<EventConsumerSelector> > ConsSelList;
    ConsSelList _eventConsumerSelectors;

  };
  
} // namespace stor

#endif // StorageManager_EventDistributor_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
