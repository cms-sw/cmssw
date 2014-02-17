// $Id: EventDistributor.cc,v 1.26 2012/04/20 10:48:02 mommsen Exp $
/// @file: EventDistributor.cc

#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/DQMEventSelector.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/ErrorStreamSelector.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventConsumerSelector.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamSelector.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

using namespace stor;


EventDistributor::EventDistributor(SharedResourcesPtr sr):
  sharedResources_(sr)
{}


EventDistributor::~EventDistributor()
{
  clearStreams();
  clearConsumers();
}

void EventDistributor::addEventToRelevantQueues( I2OChain& ioc )
{
  // special handling for faulty or incomplete events
  if ( ioc.faulty() || !ioc.complete() )
  {
    std::ostringstream msg;
    msg << "Faulty or incomplete I2OChain for event " 
      << ioc.fragmentKey().event_
      << ": 0x" << std::hex << ioc.faultyBits()
      << " received from " << ioc.hltURL()
      << " (rbBufferId " << ioc.rbBufferId() << ").";
    XCEPT_DECLARE( stor::exception::IncompleteEventMessage,
      xcept, msg.str());
    sharedResources_->alarmHandler_->
      notifySentinel(AlarmHandler::ERROR, xcept);

    DataSenderMonitorCollection& dataSenderMonColl =
      sharedResources_->statisticsReporter_->getDataSenderMonitorCollection();
    dataSenderMonColl.addFaultyEventSample(ioc);

    if ( !( sharedResources_->configuration_->getDiskWritingParams().faultyEventsStream_.empty() ) &&
      ( ioc.i2oMessageCode() == I2O_SM_DATA || ioc.i2oMessageCode() == I2O_SM_ERROR) )
      ioc.tagForStream(0); // special stream for faulty events
  }
  else
  {
    tagCompleteEventForQueues( ioc );
  }

  // Check if event belongs here at all:
  bool unexpected = true;

  if( ioc.isTaggedForAnyStream() )
  {
    unexpected = false;
    sharedResources_->streamQueue_->enqWait( ioc );
  }
  
  if( ioc.isTaggedForAnyEventConsumer() )
  {
    unexpected = false;
    sharedResources_->eventQueueCollection_->addEvent( ioc );
  }

  if( unexpected && ioc.messageCode() == Header::EVENT )
  {
    RunMonitorCollection& runMonColl =
      sharedResources_->statisticsReporter_->getRunMonitorCollection();
    runMonColl.addUnwantedEvent(ioc);
  }
}

void EventDistributor::tagCompleteEventForQueues( I2OChain& ioc )
{
  switch( ioc.messageCode() )
  {
    
    case Header::INIT:
    {
      InitMsgSharedPtr serializedProds;
      if( sharedResources_->initMsgCollection_->addIfUnique(ioc, serializedProds) )
      {
        try
        {
          InitMsgView initMsgView(&(*serializedProds)[0]);

          for_each(eventStreamSelectors_.begin(),eventStreamSelectors_.end(),
            boost::bind(&EventStreamSelector::initialize, _1, initMsgView));

          for_each(eventConsumerSelectors_.begin(), eventConsumerSelectors_.end(),
            boost::bind(&EventConsumerSelector::initialize, _1, initMsgView));
        }
        catch( stor::exception::InvalidEventSelection& e )
        {
          sharedResources_->alarmHandler_->
            notifySentinel(AlarmHandler::ERROR,e);
        }
      }
      
      DataSenderMonitorCollection& dataSenderMonColl = sharedResources_->
        statisticsReporter_->getDataSenderMonitorCollection();
      dataSenderMonColl.addInitSample(ioc);
      
      break;
    }
    
    case Header::EVENT:
    {
      for( EvtSelList::iterator it = eventStreamSelectors_.begin(),
             itEnd = eventStreamSelectors_.end();
           it != itEnd;
           ++it )
      {
        if( (*it)->acceptEvent( ioc ) )
        {
          ioc.tagForStream( (*it)->configInfo().streamId() );
        }
      }
      for( ConsSelList::iterator it = eventConsumerSelectors_.begin(),
             itEnd = eventConsumerSelectors_.end();
           it != itEnd;
           ++it )
      {
        if( (*it)->acceptEvent( ioc ) )
        {
          ioc.tagForEventConsumer( (*it)->queueId() );
        }
      }
      
      RunMonitorCollection& runMonCollection = sharedResources_->
        statisticsReporter_->getRunMonitorCollection();
      runMonCollection.getRunNumbersSeenMQ().addSampleIfLarger(ioc.runNumber());
      runMonCollection.getLumiSectionsSeenMQ().addSampleIfLarger(ioc.lumiSection());
      runMonCollection.getEventIDsReceivedMQ().addSample(ioc.eventNumber());
      
      DataSenderMonitorCollection& dataSenderMonColl = sharedResources_->
        statisticsReporter_->getDataSenderMonitorCollection();
      dataSenderMonColl.addEventSample(ioc);

      break;
    }
    
    case Header::DQM_EVENT:
    {
      utils::TimePoint_t now = utils::getCurrentTime();

      for( DQMEvtSelList::iterator it = dqmEventSelectors_.begin(),
             itEnd = dqmEventSelectors_.end();
           it != itEnd;
           ++it)
      {
        if( (*it)->acceptEvent( ioc, now ) )
        {
          ioc.tagForDQMEventConsumer( (*it)->queueId() );
        }
      }
      
      DataSenderMonitorCollection& dataSenderMonColl = sharedResources_->
        statisticsReporter_->getDataSenderMonitorCollection();
      dataSenderMonColl.addDQMEventSample(ioc);

      if( ioc.isTaggedForAnyDQMEventConsumer() )
      {
        sharedResources_->dqmEventQueue_->enqNowait( ioc );
      }
      else
      {
        sharedResources_->statisticsReporter_->getDQMEventMonitorCollection().
          getDroppedDQMEventCountsMQ().addSample(1);
      }
      
      break;
    }
    
    case Header::ERROR_EVENT:
    {
      for( ErrSelList::iterator it = errorStreamSelectors_.begin(),
             itEnd = errorStreamSelectors_.end();
           it != itEnd;
           ++it )
      {
        if( (*it)->acceptEvent( ioc ) )
        {
          ioc.tagForStream( (*it)->configInfo().streamId() );
        }
      }
      
      RunMonitorCollection& runMonCollection = sharedResources_->
        statisticsReporter_->getRunMonitorCollection();
      runMonCollection.getRunNumbersSeenMQ().addSample(ioc.runNumber());
      runMonCollection.getLumiSectionsSeenMQ().addSampleIfLarger(ioc.lumiSection());
      runMonCollection.getErrorEventIDsReceivedMQ().addSample(ioc.eventNumber());
      
      DataSenderMonitorCollection& dataSenderMonColl = sharedResources_->
        statisticsReporter_->getDataSenderMonitorCollection();
      dataSenderMonColl.addErrorEventSample(ioc);

      break;
    }
    
    default:
    {
      std::ostringstream msg;
      msg << "I2OChain with unknown message type " <<
        ioc.messageCode();
      XCEPT_DECLARE( stor::exception::WrongI2OMessageType,
                     xcept, msg.str());
      sharedResources_->
        alarmHandler_->notifySentinel(AlarmHandler::ERROR, xcept);

      // 24-Jun-2009, KAB - this is not really the best way to track this,
      // but it's probably better than nothing in the short term.
      DataSenderMonitorCollection& dataSenderMonColl = sharedResources_->
        statisticsReporter_->getDataSenderMonitorCollection();
      dataSenderMonColl.addFaultyEventSample(ioc);

      break;
    }
  }
}

const bool EventDistributor::full() const
{
  return sharedResources_->streamQueue_->full();
}


void EventDistributor::registerEventConsumer
(
  const EventConsRegPtr regPtr
)
{
  ConsSelPtr evtSel( new EventConsumerSelector(regPtr) );

  InitMsgSharedPtr initMsgPtr =
    sharedResources_->initMsgCollection_->getElementForOutputModuleLabel(
      regPtr->outputModuleLabel()
    );
  if ( initMsgPtr.get() != 0 )
  {
    uint8* initPtr = &(*initMsgPtr)[0];
    InitMsgView initView(initPtr);
    try
    {
      evtSel->initialize( initView );
    }
    catch( stor::exception::InvalidEventSelection& e )
    {
      sharedResources_->alarmHandler_->
        notifySentinel(AlarmHandler::ERROR, e);
    }
  }
  
  eventConsumerSelectors_.insert( evtSel );
}

void EventDistributor::registerDQMEventConsumer( const DQMEventConsRegPtr ptr )
{
  DQMEvtSelPtr dqmEvtSel( new DQMEventSelector(ptr) );
  dqmEventSelectors_.insert( dqmEvtSel );
}

void EventDistributor::registerEventStreams( const EvtStrConfigListPtr cl )
{
  for( EvtStrConfigList::const_iterator it = cl->begin(), itEnd = cl->end();
       it != itEnd;
       ++it )
  {
    EvtSelPtr evtSel( new EventStreamSelector(*it) );
    eventStreamSelectors_.insert( evtSel );
  }
}


void EventDistributor::registerErrorStreams( const ErrStrConfigListPtr cl )
{
  for( ErrStrConfigList::const_iterator it = cl->begin(), itEnd = cl->end();
       it != itEnd;
       ++it )
  {
    ErrSelPtr errSel( new ErrorStreamSelector(*it) );
    errorStreamSelectors_.insert( errSel );
  }
}


void EventDistributor::clearStreams()
{
  eventStreamSelectors_.clear();
  errorStreamSelectors_.clear();
}


unsigned int EventDistributor::configuredStreamCount() const
{
  return eventStreamSelectors_.size() +
    errorStreamSelectors_.size();
}


unsigned int EventDistributor::initializedStreamCount() const
{
  unsigned int counter = 0;
  for (EvtSelList::const_iterator it = eventStreamSelectors_.begin(),
         itEnd = eventStreamSelectors_.end();
       it != itEnd;
       ++it)
  {
    if ( (*it)->isInitialized() )
      ++counter;
  }
  return counter;
}


void EventDistributor::clearConsumers()
{
  eventConsumerSelectors_.clear();
  dqmEventSelectors_.clear();
}


unsigned int EventDistributor::configuredConsumerCount() const
{
  return eventConsumerSelectors_.size() + dqmEventSelectors_.size();
}


unsigned int EventDistributor::initializedConsumerCount() const
{
  unsigned int counter = 0;
  for (ConsSelList::const_iterator it = eventConsumerSelectors_.begin(),
         itEnd = eventConsumerSelectors_.end();
       it != itEnd;
       ++it)
  {
    if ( (*it)->isInitialized() )
      ++counter;
  }
  return counter;
}


void EventDistributor::checkForStaleConsumers()
{
  utils::TimePoint_t now = utils::getCurrentTime();

  EventQueueCollectionPtr eqc =
    sharedResources_->eventQueueCollection_;
  eqc->clearStaleQueues(now);

  DQMEventQueueCollectionPtr dqc =
    sharedResources_->dqmEventQueueCollection_;
  dqc->clearStaleQueues(now);
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
