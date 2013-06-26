// $Id: ConsumerMonitorCollection.cc,v 1.12 2011/03/07 15:31:32 mommsen Exp $
/// @file: ConsumerMonitorCollection.cc

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"


namespace stor {
  
  ConsumerMonitorCollection::ConsumerMonitorCollection
  (
    const utils::Duration_t& updateInterval,
    const utils::Duration_t& recentDuration
  ):
  MonitorCollection(updateInterval),
  updateInterval_(updateInterval),
  recentDuration_(recentDuration),
  totalQueuedMQ_(updateInterval, recentDuration),
  totalDroppedMQ_(updateInterval, recentDuration),
  totalServedMQ_(updateInterval, recentDuration)
  {}
  
  
  void ConsumerMonitorCollection::addQueuedEventSample
  (
    const QueueID& qid,
    const unsigned int& data_size
  )
  {
    boost::mutex::scoped_lock l( mutex_ );
    addEventSampleToMap(qid, data_size, qmap_);
    totalQueuedMQ_.addSample(data_size);
  }
  
  
  void ConsumerMonitorCollection::addDroppedEvents
  (
    const QueueID& qid,
    const size_t& count
  )
  {
    boost::mutex::scoped_lock l( mutex_ );
    addEventSampleToMap(qid, count, dmap_);
    totalDroppedMQ_.addSample(count);
  }
  
  
  void ConsumerMonitorCollection::addServedEventSample
  (
    const QueueID& qid,
    const unsigned int& data_size
  )
  {
    boost::mutex::scoped_lock l( mutex_ );
    addEventSampleToMap(qid, data_size, smap_);
    totalServedMQ_.addSample(data_size);
  }
  
  
  void ConsumerMonitorCollection::addEventSampleToMap
  (
    const QueueID& qid,
    const unsigned int& data_size,
    ConsStatMap& map
  )
  {
    ConsStatMap::iterator pos = map.lower_bound(qid);
    
    // 05-Oct-2009, KAB - added a test of whether qid appears before pos->first
    // in the map sort order.  Since lower_bound can return a non-end iterator
    // even when qid is not in the map, we need to complete the test of whether
    // qid is in the map.  (Another way to look at this is we need to implement
    // the full test described in the efficientAddOrUpdates pattern suggested
    // by Item 24 of 'Effective STL' by Scott Meyers.)
    if (pos == map.end() || (map.key_comp()(qid, pos->first)))
    {
      // The key does not exist in the map, add it to the map
      // Use pos as a hint to insert, so it can avoid another lookup
      pos = map.insert(pos,
        ConsStatMap::value_type(qid, 
          MonitoredQuantityPtr(
            new MonitoredQuantity(updateInterval_, recentDuration_)
          )
        )
      );
    }
    
    pos->second->addSample( data_size );
  }
  
  
  bool ConsumerMonitorCollection::getQueued
  (
    const QueueID& qid,
    MonitoredQuantity::Stats& result
  ) const
  {
    boost::mutex::scoped_lock l( mutex_ );
    return getValueFromMap( qid, result, qmap_ );
  }
  
  
  bool ConsumerMonitorCollection::getServed
  (
    const QueueID& qid,
    MonitoredQuantity::Stats& result
  ) const
  {
    boost::mutex::scoped_lock l( mutex_ );
    return getValueFromMap( qid, result, smap_ );
  }
  
  
  bool ConsumerMonitorCollection::getDropped
  (
    const QueueID& qid,
    MonitoredQuantity::Stats& result
  ) const
  {
    boost::mutex::scoped_lock l( mutex_ );
    return getValueFromMap( qid, result, dmap_ );
  }
  
  bool ConsumerMonitorCollection::getValueFromMap
  (
    const QueueID& qid,
    MonitoredQuantity::Stats& result,
    const ConsStatMap& map
  ) const
  {
    ConsStatMap::const_iterator pos = map.find(qid);
    
    if (pos == map.end()) return false;
    
    pos->second->getStats( result );
    return true;
  }
  
  
  void ConsumerMonitorCollection::getTotalStats( TotalStats& totalStats ) const
  {
    totalQueuedMQ_.getStats(totalStats.queuedStats);
    totalDroppedMQ_.getStats(totalStats.droppedStats);
    totalServedMQ_.getStats(totalStats.servedStats);
  }
  
  void ConsumerMonitorCollection::resetCounters()
  {
    boost::mutex::scoped_lock l( mutex_ );
    for( ConsStatMap::iterator i = qmap_.begin(); i != qmap_.end(); ++i )
      i->second->reset();
    for( ConsStatMap::iterator i = smap_.begin(); i != smap_.end(); ++i )
      i->second->reset();
    for( ConsStatMap::iterator i = dmap_.begin(); i != dmap_.end(); ++i )
      i->second->reset();
    
    totalQueuedMQ_.reset();
    totalDroppedMQ_.reset();
    totalServedMQ_.reset();
  }
  
  
  void ConsumerMonitorCollection::do_calculateStatistics()
  {
    boost::mutex::scoped_lock l( mutex_ );
    for( ConsStatMap::iterator i = qmap_.begin(); i != qmap_.end(); ++i )
      i->second->calculateStatistics();
    for( ConsStatMap::iterator i = smap_.begin(); i != smap_.end(); ++i )
      i->second->calculateStatistics();
    for( ConsStatMap::iterator i = dmap_.begin(); i != dmap_.end(); ++i )
      i->second->calculateStatistics();
    
    totalQueuedMQ_.calculateStatistics();
    totalDroppedMQ_.calculateStatistics();
    totalServedMQ_.calculateStatistics();
  }
  
  
  void ConsumerMonitorCollection::do_reset()
  {
    boost::mutex::scoped_lock l( mutex_ );
    qmap_.clear();
    smap_.clear();
    dmap_.clear();
    
    totalQueuedMQ_.reset();
    totalDroppedMQ_.reset();
    totalServedMQ_.reset();
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
