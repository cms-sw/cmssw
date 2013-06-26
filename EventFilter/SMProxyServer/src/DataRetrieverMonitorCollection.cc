// $Id: DataRetrieverMonitorCollection.cc,v 1.3 2011/05/09 11:03:34 mommsen Exp $
/// @file: DataRetrieverMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include <boost/pointer_cast.hpp>

#include "EventFilter/SMProxyServer/interface/Exception.h"
#include "EventFilter/SMProxyServer/interface/DataRetrieverMonitorCollection.h"


namespace smproxy {
  
  DataRetrieverMonitorCollection::DataRetrieverMonitorCollection
  (
    const stor::utils::Duration_t& updateInterval,
    stor::AlarmHandlerPtr alarmHandler
  ) :
  MonitorCollection(updateInterval),
  updateInterval_(updateInterval),
  alarmHandler_(alarmHandler),
  totals_(updateInterval),
  eventTypeMqMap_(updateInterval)
  {}
  
  
  ConnectionID DataRetrieverMonitorCollection::addNewConnection
  (
    const stor::RegPtr regPtr
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    ++nextConnectionId_;
    
    DataRetrieverMQPtr dataRetrieverMQ( new DataRetrieverMQ(regPtr, updateInterval_) );
    retrieverMqMap_.insert(
      RetrieverMqMap::value_type(nextConnectionId_, dataRetrieverMQ)
    );
    
    eventTypeMqMap_.insert(regPtr);
    
    connectionMqMap_.insert(ConnectionMqMap::value_type(
        regPtr->sourceURL(),
        EventMQPtr(new EventMQ(updateInterval_))
      ));
    
    return nextConnectionId_;
  }
  
  
  bool DataRetrieverMonitorCollection::setConnectionStatus
  (
    const ConnectionID& connectionId,
    const ConnectionStatus& status
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    RetrieverMqMap::const_iterator pos = retrieverMqMap_.find(connectionId);
    if ( pos == retrieverMqMap_.end() ) return false;
    pos->second->connectionStatus_ = status;
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::getEventTypeStatsForConnection
  (
    const ConnectionID& connectionId,
    EventTypePerConnectionStats& stats
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    RetrieverMqMap::const_iterator pos = retrieverMqMap_.find(connectionId);
    
    if ( pos == retrieverMqMap_.end() ) return false;
    
    stats.regPtr = pos->second->regPtr_;
    stats.connectionStatus = pos->second->connectionStatus_;
    pos->second->eventMQ_->getStats(stats.eventStats);
    
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::addRetrievedSample
  (
    const ConnectionID& connectionId,
    const unsigned int& size
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    
    RetrieverMqMap::const_iterator retrieverPos = retrieverMqMap_.find(connectionId);
    if ( retrieverPos == retrieverMqMap_.end() ) return false;
    
    const double sizeKB = static_cast<double>(size) / 1024;
    retrieverPos->second->eventMQ_->size_.addSample(sizeKB);
    
    const stor::RegPtr regPtr = retrieverPos->second->regPtr_;
    
    eventTypeMqMap_.addSample(regPtr, sizeKB);
    
    const std::string sourceURL = regPtr->sourceURL();
    ConnectionMqMap::const_iterator connectionPos = connectionMqMap_.find(sourceURL);
    connectionPos->second->size_.addSample(sizeKB);
    
    totals_.size_.addSample(sizeKB);
    
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::receivedCorruptedEvent
  (
    const ConnectionID& connectionId
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    
    RetrieverMqMap::const_iterator retrieverPos = retrieverMqMap_.find(connectionId);
    if ( retrieverPos == retrieverMqMap_.end() ) return false;
    
    retrieverPos->second->eventMQ_->corruptedEvents_.addSample(1);
    
    const stor::RegPtr regPtr = retrieverPos->second->regPtr_;
    
    eventTypeMqMap_.receivedCorruptedEvent(regPtr);
    
    const std::string sourceURL = regPtr->sourceURL();
    ConnectionMqMap::const_iterator connectionPos = connectionMqMap_.find(sourceURL);
    connectionPos->second->corruptedEvents_.addSample(1);
    
    totals_.corruptedEvents_.addSample(1);

    return true;
  }
  
  
  void DataRetrieverMonitorCollection::getSummaryStats(SummaryStats& stats) const
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    
    stats.registeredSMs = 0;
    stats.activeSMs = 0;
    
    for (RetrieverMqMap::const_iterator it = retrieverMqMap_.begin(),
           itEnd = retrieverMqMap_.end(); it != itEnd; ++it)
    {
      ++stats.registeredSMs;
      if ( it->second->connectionStatus_ == CONNECTED )
        ++stats.activeSMs;
    }
    
    eventTypeMqMap_.getStats(stats.eventTypeStats);
    
    totals_.getStats(stats.totals);
  }
  
  
  void DataRetrieverMonitorCollection::getStatsByConnection(ConnectionStats& cs) const
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    cs.clear();
    
    for (ConnectionMqMap::const_iterator it = connectionMqMap_.begin(),
           itEnd = connectionMqMap_.end(); it != itEnd; ++it)
    {
      EventStats stats;
      it->second->getStats(stats);
      cs.insert(ConnectionStats::value_type(it->first, stats));
    }
  }
  
  
  void DataRetrieverMonitorCollection::getStatsByEventTypesPerConnection
  (
    EventTypePerConnectionStatList& etsl
  ) const
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    etsl.clear();
    
    for (RetrieverMqMap::const_iterator it = retrieverMqMap_.begin(),
           itEnd = retrieverMqMap_.end(); it != itEnd; ++it)
    {
      const DataRetrieverMQPtr mq = it->second;
      EventTypePerConnectionStats stats;
      stats.regPtr = mq->regPtr_;
      stats.connectionStatus = mq->connectionStatus_;
      mq->eventMQ_->getStats(stats.eventStats);
      etsl.push_back(stats);
    }
    std::sort(etsl.begin(), etsl.end());
  }
  
  
  void DataRetrieverMonitorCollection::configureAlarms(AlarmParams const& alarmParams)
  {
    alarmParams_ = alarmParams;
  }
  
  
  void DataRetrieverMonitorCollection::sendAlarms()
  {
    if ( ! alarmParams_.sendAlarms_ ) return;
    
    checkForCorruptedEvents();
  }
  
  
  void DataRetrieverMonitorCollection::checkForCorruptedEvents()
  {
    const std::string alarmName = "CorruptedEvents";
    
    EventStats eventStats;
    totals_.getStats(eventStats);
    const double corruptedEventRate =
      eventStats.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::RECENT);
    if ( corruptedEventRate > alarmParams_.corruptedEventRate_ )
    {
      std::ostringstream msg;
      msg << "Received " << corruptedEventRate << " Hz of corrupted events from StorageManagers.";
      XCEPT_DECLARE(exception::CorruptedEvents, ex, msg.str());
      alarmHandler_->raiseAlarm(alarmName, stor::AlarmHandler::ERROR, ex);
    }
    else if ( corruptedEventRate < (alarmParams_.corruptedEventRate_ * 0.9) )
      // avoid revoking the alarm if we're close to the limit
    {
      alarmHandler_->revokeAlarm(alarmName);
    }
  }
  
  
  void DataRetrieverMonitorCollection::do_calculateStatistics()
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    
    totals_.calculateStatistics();
    
    for (RetrieverMqMap::const_iterator it = retrieverMqMap_.begin(),
           itEnd = retrieverMqMap_.end(); it != itEnd; ++it)
    {
      it->second->eventMQ_->calculateStatistics();
    }
    
    for (ConnectionMqMap::const_iterator it = connectionMqMap_.begin(),
           itEnd = connectionMqMap_.end(); it != itEnd; ++it)
    {
      it->second->calculateStatistics();
    }
    
    eventTypeMqMap_.calculateStatistics();

    sendAlarms();
  }
  
  
  void DataRetrieverMonitorCollection::do_reset()
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    totals_.reset();
    retrieverMqMap_.clear();
    connectionMqMap_.clear();
    eventTypeMqMap_.clear();
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  insert(const stor::RegPtr consumer)
  {
    return (
      insert(boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(consumer)) ||
      insert(boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(consumer))
    );
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  addSample(const stor::RegPtr consumer, const double& sizeKB)
  {
    return (
      addSample(boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(consumer), sizeKB) ||
      addSample(boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(consumer), sizeKB)
    );
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  receivedCorruptedEvent(const stor::RegPtr consumer)
  {
    return (
      receivedCorruptedEvent(boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(consumer)) ||
      receivedCorruptedEvent(boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(consumer))
    );
  }
  
  
  void DataRetrieverMonitorCollection::EventTypeMqMap::
  getStats(SummaryStats::EventTypeStatList& eventTypeStats) const
  {
    eventTypeStats.clear();
    eventTypeStats.reserve(eventMap_.size()+dqmEventMap_.size());
    
    for (EventMap::const_iterator it = eventMap_.begin(),
           itEnd = eventMap_.end(); it != itEnd; ++it)
    {
      EventStats eventStats;
      it->second->size_.getStats(eventStats.sizeStats);
      it->second->corruptedEvents_.getStats(eventStats.corruptedEventsStats);
      eventTypeStats.push_back(
        std::make_pair(it->first, eventStats));
    }
    
    for (DQMEventMap::const_iterator it = dqmEventMap_.begin(),
           itEnd = dqmEventMap_.end(); it != itEnd; ++it)
    {
      EventStats eventStats;
      it->second->size_.getStats(eventStats.sizeStats);
      it->second->corruptedEvents_.getStats(eventStats.corruptedEventsStats);
      eventTypeStats.push_back(
        std::make_pair(it->first, eventStats));
    }
  }
  
  
  void DataRetrieverMonitorCollection::EventTypeMqMap::
  calculateStatistics()
  {
    for (EventMap::iterator it = eventMap_.begin(),
           itEnd = eventMap_.end(); it != itEnd; ++it)
    {
      it->second->size_.calculateStatistics();
      it->second->corruptedEvents_.calculateStatistics();
    }
    for (DQMEventMap::iterator it = dqmEventMap_.begin(),
           itEnd = dqmEventMap_.end(); it != itEnd; ++it)
    {
      it->second->size_.calculateStatistics();
      it->second->corruptedEvents_.calculateStatistics();
    }
  }
  
  
  void DataRetrieverMonitorCollection::EventTypeMqMap::
  clear()
  {
    eventMap_.clear();
    dqmEventMap_.clear();
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  insert(const stor::EventConsRegPtr eventConsumer)
  {
    if ( eventConsumer == 0 ) return false;
    eventMap_.insert(EventMap::value_type(eventConsumer,
        EventMQPtr( new EventMQ(updateInterval_) )
      ));
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  insert(const stor::DQMEventConsRegPtr dqmEventConsumer)
  {
    if ( dqmEventConsumer == 0 ) return false;
    dqmEventMap_.insert(DQMEventMap::value_type(dqmEventConsumer,
        EventMQPtr( new EventMQ(updateInterval_) )
      ));
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  addSample(const stor::EventConsRegPtr eventConsumer, const double& sizeKB)
  {
    if ( eventConsumer == 0 ) return false;
    EventMap::const_iterator pos = eventMap_.find(eventConsumer);
    pos->second->size_.addSample(sizeKB);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  addSample(const stor::DQMEventConsRegPtr dqmEventConsumer, const double& sizeKB)
  {
    if ( dqmEventConsumer == 0 ) return false;
    DQMEventMap::const_iterator pos = dqmEventMap_.find(dqmEventConsumer);
    pos->second->size_.addSample(sizeKB);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  receivedCorruptedEvent(const stor::EventConsRegPtr eventConsumer)
  {
    if ( eventConsumer == 0 ) return false;
    EventMap::const_iterator pos = eventMap_.find(eventConsumer);
    pos->second->corruptedEvents_.addSample(1);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  receivedCorruptedEvent(const stor::DQMEventConsRegPtr dqmEventConsumer)
  {
    if ( dqmEventConsumer == 0 ) return false;
    DQMEventMap::const_iterator pos = dqmEventMap_.find(dqmEventConsumer);
    pos->second->corruptedEvents_.addSample(1);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypePerConnectionStats::
  operator<(const EventTypePerConnectionStats& other) const
  {
    if ( regPtr->sourceURL() != other.regPtr->sourceURL() )
      return ( regPtr->sourceURL() < other.regPtr->sourceURL() );
    
    stor::EventConsRegPtr ecrp =
      boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(regPtr);
    stor::EventConsRegPtr ecrpOther =
      boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(other.regPtr);
    if ( ecrp && ecrpOther )
      return ( *ecrp < *ecrpOther);
    
    stor::DQMEventConsRegPtr dcrp =
      boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(regPtr);
    stor::DQMEventConsRegPtr dcrpOther =
      boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(other.regPtr);
    if ( dcrp && dcrpOther )
      return ( *dcrp < *dcrpOther);
    
    return false;
  }
  
  
  DataRetrieverMonitorCollection::EventMQ::EventMQ
  (
    const stor::utils::Duration_t& updateInterval
  ):
  size_(updateInterval, boost::posix_time::seconds(60)),
  corruptedEvents_(updateInterval, boost::posix_time::seconds(60))
  {}


  void DataRetrieverMonitorCollection::EventMQ::getStats(EventStats& stats) const
  {
    size_.getStats(stats.sizeStats);
    corruptedEvents_.getStats(stats.corruptedEventsStats);
  }
  
  
  void DataRetrieverMonitorCollection::EventMQ::calculateStatistics()
  {
    size_.calculateStatistics();
    corruptedEvents_.calculateStatistics();
  }
  
  
  void DataRetrieverMonitorCollection::EventMQ::reset()
  {
    size_.reset();
    corruptedEvents_.reset();
  }
  
  
  DataRetrieverMonitorCollection::DataRetrieverMQ::DataRetrieverMQ
  (
    const stor::RegPtr regPtr,
    const stor::utils::Duration_t& updateInterval
  ):
  regPtr_(regPtr),
  connectionStatus_(UNKNOWN),
  eventMQ_(new EventMQ(updateInterval))
  {}
  
} // namespace smproxy


std::ostream& smproxy::operator<<
(
  std::ostream& os,
  const DataRetrieverMonitorCollection::ConnectionStatus& status
)
{
  switch (status)
  {
    case DataRetrieverMonitorCollection::CONNECTED :
      os << "Connected";
      break;
    case DataRetrieverMonitorCollection::CONNECTION_FAILED :
      os << "Could not connect. SM not running?";
      break;
    case DataRetrieverMonitorCollection::DISCONNECTED :
      os << "Lost connection to SM. Did it fail?";
      break;
    case DataRetrieverMonitorCollection::UNKNOWN :
      os << "unknown";
      break;
  }
  
  return os;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
