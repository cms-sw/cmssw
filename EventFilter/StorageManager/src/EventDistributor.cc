// $Id: EventDistributor.cc,v 1.3 2009/06/24 19:11:22 biery Exp $
/// @file: EventDistributor.cc

#include "EventFilter/StorageManager/interface/EventDistributor.h"

using namespace stor;


EventDistributor::EventDistributor(SharedResourcesPtr sr):
  _sharedResources(sr)
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
    // mark these events for the special SM error stream
    
    // log a warning???

    DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
      _statisticsReporter->getDataSenderMonitorCollection();
    dataSenderMonColl.addStaleChainSample(ioc);
  }
  else
  {
    tagCompleteEventForQueues( ioc );
  }
  
  if( ioc.isTaggedForAnyStream() )
  {
    _sharedResources->_streamQueue->enq_wait( ioc );
  }
  
  if( ioc.isTaggedForAnyEventConsumer() )
  {
    _sharedResources->_eventConsumerQueueCollection->addEvent( ioc );
  }
}

void EventDistributor::tagCompleteEventForQueues( I2OChain& ioc )
{
  switch( ioc.messageCode() )
  {
    
    case Header::INIT:
    {
      std::vector<unsigned char> b;
      ioc.copyFragmentsIntoBuffer(b);
      InitMsgView imv( &b[0] );
      if( _sharedResources->_initMsgCollection->addIfUnique( imv ) )
      {
        for( EvtSelList::iterator it = _eventStreamSelectors.begin(),
               itEnd = _eventStreamSelectors.end();
             it != itEnd;
             ++it )
        {
          it->initialize( imv );
        }
        for( ConsSelList::iterator it = _eventConsumerSelectors.begin(),
               itEnd = _eventConsumerSelectors.end();
             it != itEnd;
             ++it )
        {
          it->initialize( imv );
        }
      }
      
      DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
        _statisticsReporter->getDataSenderMonitorCollection();
      dataSenderMonColl.addInitSample(ioc);
      
      break;
    }
    
    case Header::EVENT:
    {
      for( EvtSelList::iterator it = _eventStreamSelectors.begin(),
             itEnd = _eventStreamSelectors.end();
           it != itEnd;
           ++it )
      {
        if( it->acceptEvent( ioc ) )
        {
          ioc.tagForStream( it->configInfo().streamId() );
        }
      }
      for( ConsSelList::iterator it = _eventConsumerSelectors.begin(),
             itEnd = _eventConsumerSelectors.end();
           it != itEnd;
           ++it )
      {
        if( it->acceptEvent( ioc ) )
        {
          ioc.tagForEventConsumer( it->queueId() );
        }
      }
      
      RunMonitorCollection& runMonCollection = _sharedResources->
        _statisticsReporter->getRunMonitorCollection();
      runMonCollection.getRunNumbersSeenMQ().addSample(ioc.runNumber());
      runMonCollection.getLumiSectionsSeenMQ().addSample(ioc.lumiSection());
      runMonCollection.getEventIDsReceivedMQ().addSample(ioc.eventNumber());
      
      DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
        _statisticsReporter->getDataSenderMonitorCollection();
      dataSenderMonColl.addEventSample(ioc);

      break;
    }
    
    case Header::DQM_EVENT:
    {
      for( DQMEvtSelList::iterator it = _dqmEventSelectors.begin(),
             itEnd = _dqmEventSelectors.end();
           it != itEnd;
           ++it)
      {
        if( it->acceptEvent( ioc ) )
        {
          ioc.tagForDQMEventConsumer( it->queueId() );
        }
      }
      
      // Pass any DQM event to the DQM event processor, as it might write 
      // DQM histograms to disk which are not requested by any consumer
      // Put this here or in EventDistributor::addEventToRelevantQueues?
      _sharedResources->_dqmEventQueue->enq_nowait( ioc );
      
      DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
        _statisticsReporter->getDataSenderMonitorCollection();
      dataSenderMonColl.addDQMEventSample(ioc);

      break;
    }
    
    case Header::ERROR_EVENT:
    {
      for( ErrSelList::iterator it = _errorStreamSelectors.begin(),
             itEnd = _errorStreamSelectors.end();
           it != itEnd;
           ++it )
      {
        if( it->acceptEvent( ioc ) )
        {
          ioc.tagForStream( it->configInfo().streamId() );
        }
      }
      
      RunMonitorCollection& runMonCollection = _sharedResources->
        _statisticsReporter->getRunMonitorCollection();
      runMonCollection.getRunNumbersSeenMQ().addSample(ioc.runNumber());
      runMonCollection.getLumiSectionsSeenMQ().addSample(ioc.lumiSection());
      runMonCollection.getErrorEventIDsReceivedMQ().addSample(ioc.eventNumber());
      
      DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
        _statisticsReporter->getDataSenderMonitorCollection();
      dataSenderMonColl.addErrorEventSample(ioc);

      break;
    }
    
    default:
    {
      // Log error and/or go to failed state???


      // 24-Jun-2009, KAB - this is not really the best way to track this,
      // but it's probably better than nothing in the short term.
      DataSenderMonitorCollection& dataSenderMonColl = _sharedResources->
        _statisticsReporter->getDataSenderMonitorCollection();
      dataSenderMonColl.addStaleChainSample(ioc);

      break;
    }
  }
}

const bool EventDistributor::full() const
{
  return _sharedResources->_streamQueue->full();
}


void EventDistributor::registerEventConsumer
(
  const EventConsumerRegistrationInfo* registrationInfo
)
{
  EventConsumerSelector evtSel( registrationInfo );

  InitMsgSharedPtr initMsgPtr =
    _sharedResources->_initMsgCollection->getElementForOutputModule( registrationInfo->selHLTOut() );
  if ( initMsgPtr.get() != 0 )
  {
    uint8* regPtr = &(*initMsgPtr)[0];
    InitMsgView initView(regPtr);
    evtSel.initialize( initView );
  }
  
  _eventConsumerSelectors.push_back( evtSel );
}

void EventDistributor::registerDQMEventConsumer( const DQMEventConsumerRegistrationInfo* ptr )
{
  _dqmEventSelectors.push_back( DQMEventSelector( ptr ) );
}

void EventDistributor::registerEventStreams( const EvtStrConfigListPtr cl )
{
  for( EvtStrConfigList::const_iterator it = cl->begin(), itEnd = cl->end();
       it != itEnd;
       ++it )
  {
    _eventStreamSelectors.push_back( EventStreamSelector( *it ) );
  }
}


void EventDistributor::registerErrorStreams( const ErrStrConfigListPtr cl )
{
  for( ErrStrConfigList::const_iterator it = cl->begin(), itEnd = cl->end();
       it != itEnd;
       ++it )
  {
    _errorStreamSelectors.push_back( ErrorStreamSelector( *it ) );
  }
}


void EventDistributor::clearStreams()
{
  _eventStreamSelectors.clear();
  _errorStreamSelectors.clear();
}


unsigned int EventDistributor::configuredStreamCount() const
{
  return _eventStreamSelectors.size() +
    _errorStreamSelectors.size();
}


unsigned int EventDistributor::initializedStreamCount() const
{
  unsigned int counter = 0;
  for (EvtSelList::const_iterator it = _eventStreamSelectors.begin(),
         itEnd = _eventStreamSelectors.end();
       it != itEnd;
       ++it)
  {
    if ( it->isInitialized() )
      ++counter;
  }
  return counter;
}


void EventDistributor::clearConsumers()
{
  _eventConsumerSelectors.clear();
  _dqmEventSelectors.clear();
}


unsigned int EventDistributor::configuredConsumerCount() const
{
  return _eventConsumerSelectors.size() + _dqmEventSelectors.size();
}


unsigned int EventDistributor::initializedConsumerCount() const
{
  unsigned int counter = 0;
  for (ConsSelList::const_iterator it = _eventConsumerSelectors.begin(),
         itEnd = _eventConsumerSelectors.end();
       it != itEnd;
       ++it)
  {
    if ( it->isInitialized() )
      ++counter;
  }
  return counter;
}


void EventDistributor::checkForStaleConsumers()
{

  /////////////////////
  // event consumers //
  /////////////////////

  std::vector<QueueID> stale_qs;
  _sharedResources->_eventConsumerQueueCollection->clearStaleQueues( stale_qs );

  RegistrationCollection::ConsumerRegistrations cregs;
  _sharedResources->_registrationCollection->getEventConsumers( cregs );

  // Double linear search, should try to optimize...
  for( ConsSelList::iterator i = _eventConsumerSelectors.begin();
           i != _eventConsumerSelectors.end(); ++i )
    {

      // First, assume the consumer is active. If we find its queue in
      // the stale list, we'll mark it stale in the inner loop.
      i->markAsActive();

      for( std::vector<QueueID>::const_iterator j = stale_qs.begin();
           j != stale_qs.end(); ++j )
        {
          if( i->queueId() == *j )
            {
              i->markAsStale();
            }
        }

      // Finally, to make matters even worse, we iterate over the
      // registrations to set the staleness flags so that it can be
      // displayed on the web page:
      for( RegistrationCollection::ConsumerRegistrations::iterator k = cregs.begin();
           k != cregs.end(); ++k )
        {
          if( (*k)->queueId() == i->queueId() )
            {
              (*k)->setStaleness( i->isStale() );
            }
        }

    }
  
  ///////////////////
  // dqm consumers //
  ///////////////////

  stale_qs.clear();
  _sharedResources->_dqmEventConsumerQueueCollection->clearStaleQueues( stale_qs );

  RegistrationCollection::DQMConsumerRegistrations dqm_cregs;
  _sharedResources->_registrationCollection->getDQMEventConsumers( dqm_cregs );

  // Double linear search, should try to optimize...
  for( DQMEvtSelList::iterator i = _dqmEventSelectors.begin();
           i != _dqmEventSelectors.end(); ++i )
    {

      // First, assume the consumer is active. If we find its queue in
      // the stale list, we'll mark it stale in the inner loop.
      i->markAsActive();

      for( std::vector<QueueID>::const_iterator j = stale_qs.begin();
           j != stale_qs.end(); ++j )
        {
          if( i->queueId() == *j )
            {
              i->markAsStale();
            }
        }

      // Finally, to make matters even worse, we iterate over the
      // registrations to set the staleness flags so that it can be
      // displayed on the web page:
      for( RegistrationCollection::DQMConsumerRegistrations::iterator k = dqm_cregs.begin();
           k != dqm_cregs.end(); ++k )
        {
          if( (*k)->queueId() == i->queueId() )
            {
              (*k)->setStaleness( i->isStale() );
            }
        }

    }

}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
