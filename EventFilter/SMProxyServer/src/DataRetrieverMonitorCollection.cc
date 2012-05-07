// $Id: DataRetrieverMonitorCollection.cc,v 1.1.2.8 2011/03/01 08:32:15 mommsen Exp $
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
    const stor::utils::Duration_t& updateInterval
  ) :
  MonitorCollection(updateInterval),
  totalSize_(updateInterval, boost::posix_time::seconds(60)),
  updateInterval_(updateInterval),
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
    
    connectionMqMap_.insert(ConnectionMqMap::value_type(regPtr->sourceURL(),
        stor::MonitoredQuantityPtr(
          new stor::MonitoredQuantity(updateInterval_, boost::posix_time::seconds(60))
        )
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
    EventTypeStats& stats
  )
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    RetrieverMqMap::const_iterator pos = retrieverMqMap_.find(connectionId);
    
    if ( pos == retrieverMqMap_.end() ) return false;
    
    stats.regPtr = pos->second->regPtr_;
    stats.connectionStatus = pos->second->connectionStatus_;
    pos->second->size_.getStats(stats.sizeStats);
    
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
    retrieverPos->second->size_.addSample(sizeKB);
    
    const stor::RegPtr regPtr = retrieverPos->second->regPtr_;
    
    eventTypeMqMap_.addSample(regPtr, sizeKB);
    
    const std::string sourceURL = regPtr->sourceURL();
    ConnectionMqMap::const_iterator connectionPos = connectionMqMap_.find(sourceURL);
    connectionPos->second->addSample(sizeKB);
    
    totalSize_.addSample(sizeKB);
    
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
    
    totalSize_.getStats(stats.sizeStats);
  }
  
  
  void DataRetrieverMonitorCollection::getStatsByConnection(ConnectionStats& cs) const
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    cs.clear();
    
    for (ConnectionMqMap::const_iterator it = connectionMqMap_.begin(),
           itEnd = connectionMqMap_.end(); it != itEnd; ++it)
    {
      stor::MonitoredQuantity::Stats stats;
      it->second->getStats(stats);
      cs.insert(ConnectionStats::value_type(it->first, stats));
    }
  }
  
  
  void DataRetrieverMonitorCollection::getStatsByEventTypes(EventTypeStatList& etsl) const
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    etsl.clear();
    
    for (RetrieverMqMap::const_iterator it = retrieverMqMap_.begin(),
           itEnd = retrieverMqMap_.end(); it != itEnd; ++it)
    {
      const DataRetrieverMQPtr mq = it->second;
      EventTypeStats stats;
      stats.regPtr = mq->regPtr_;
      stats.connectionStatus = mq->connectionStatus_;
      mq->size_.getStats(stats.sizeStats);
      etsl.push_back(stats);
    }
    std::sort(etsl.begin(), etsl.end());
  }
  
  
  void DataRetrieverMonitorCollection::do_calculateStatistics()
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    
    totalSize_.calculateStatistics();
    
    for (RetrieverMqMap::const_iterator it = retrieverMqMap_.begin(),
           itEnd = retrieverMqMap_.end(); it != itEnd; ++it)
    {
      it->second->size_.calculateStatistics();
    }
    
    for (ConnectionMqMap::const_iterator it = connectionMqMap_.begin(),
           itEnd = connectionMqMap_.end(); it != itEnd; ++it)
    {
      it->second->calculateStatistics();
    }
    
    eventTypeMqMap_.calculateStatistics();
  }
  
  
  void DataRetrieverMonitorCollection::do_reset()
  {
    boost::mutex::scoped_lock sl(statsMutex_);
    totalSize_.reset();
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
  
  
  void DataRetrieverMonitorCollection::EventTypeMqMap::
  getStats(SummaryStats::EventTypeStatList& eventTypeStats) const
  {
    eventTypeStats.clear();
    eventTypeStats.reserve(eventMap_.size()+dqmEventMap_.size());
    
    for (EventMap::const_iterator it = eventMap_.begin(),
           itEnd = eventMap_.end(); it != itEnd; ++it)
    {
      stor::MonitoredQuantity::Stats etStats;
      it->second->getStats(etStats);
      eventTypeStats.push_back(
        std::make_pair(it->first, etStats));
    }
    
    for (DQMEventMap::const_iterator it = dqmEventMap_.begin(),
           itEnd = dqmEventMap_.end(); it != itEnd; ++it)
    {
      stor::MonitoredQuantity::Stats etStats;
      it->second->getStats(etStats);
      eventTypeStats.push_back(
        std::make_pair(it->first, etStats));
    }
  }
  
  
  void DataRetrieverMonitorCollection::EventTypeMqMap::
  calculateStatistics()
  {
    for (EventMap::iterator it = eventMap_.begin(),
           itEnd = eventMap_.end(); it != itEnd; ++it)
    {
      it->second->calculateStatistics();
    }
    for (DQMEventMap::iterator it = dqmEventMap_.begin(),
           itEnd = dqmEventMap_.end(); it != itEnd; ++it)
    {
      it->second->calculateStatistics();
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
        stor::MonitoredQuantityPtr(
          new stor::MonitoredQuantity( updateInterval_, boost::posix_time::seconds(60) )
        )));
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  insert(const stor::DQMEventConsRegPtr dqmEventConsumer)
  {
    if ( dqmEventConsumer == 0 ) return false;
    dqmEventMap_.insert(DQMEventMap::value_type(dqmEventConsumer,
        stor::MonitoredQuantityPtr(
          new stor::MonitoredQuantity( updateInterval_, boost::posix_time::seconds(60) )
        )));
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  addSample(const stor::EventConsRegPtr eventConsumer, const double& sizeKB)
  {
    if ( eventConsumer == 0 ) return false;
    EventMap::const_iterator pos = eventMap_.find(eventConsumer);
    pos->second->addSample(sizeKB);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeMqMap::
  addSample(const stor::DQMEventConsRegPtr dqmEventConsumer, const double& sizeKB)
  {
    if ( dqmEventConsumer == 0 ) return false;
    DQMEventMap::const_iterator pos = dqmEventMap_.find(dqmEventConsumer);
    pos->second->addSample(sizeKB);
    return true;
  }
  
  
  bool DataRetrieverMonitorCollection::EventTypeStats::operator<(const EventTypeStats& other) const
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
  
  
  DataRetrieverMonitorCollection::DataRetrieverMQ::DataRetrieverMQ
  (
    const stor::RegPtr regPtr,
    const stor::utils::Duration_t& updateInterval
  ):
  regPtr_(regPtr),
  connectionStatus_(UNKNOWN),
  size_(updateInterval, boost::posix_time::seconds(60))
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
