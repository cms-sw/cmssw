// $Id: RunMonitorCollection.cc,v 1.18 2012/06/08 10:20:33 mommsen Exp $
/// @file: RunMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <boost/bind.hpp>

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"


namespace stor {
  
  RunMonitorCollection::RunMonitorCollection
  (
    const utils::Duration_t& updateInterval,
    SharedResourcesPtr sr
  ) :
  MonitorCollection(updateInterval),
  eventIDsReceived_(updateInterval, boost::posix_time::seconds(1)),
  errorEventIDsReceived_(updateInterval, boost::posix_time::seconds(1)),
  unwantedEventIDsReceived_(updateInterval, boost::posix_time::seconds(1)),
  runNumbersSeen_(updateInterval, boost::posix_time::seconds(1)),
  lumiSectionsSeen_(updateInterval, boost::posix_time::seconds(1)),
  eolsSeen_(updateInterval, boost::posix_time::seconds(1)),
  sharedResources_(sr)
  {}
  
  
  void RunMonitorCollection::configureAlarms(AlarmParams const& alarmParams)
  {
    alarmParams_ = alarmParams;
  }
  
  
  void RunMonitorCollection::do_calculateStatistics()
  {
    eventIDsReceived_.calculateStatistics();
    errorEventIDsReceived_.calculateStatistics();
    unwantedEventIDsReceived_.calculateStatistics();
    runNumbersSeen_.calculateStatistics();
    lumiSectionsSeen_.calculateStatistics();
    eolsSeen_.calculateStatistics();
    
    checkForBadEvents();
  }
  
  
  void RunMonitorCollection::do_reset()
  {
    eventIDsReceived_.reset();
    errorEventIDsReceived_.reset();
    unwantedEventIDsReceived_.reset();
    runNumbersSeen_.reset();
    lumiSectionsSeen_.reset();
    eolsSeen_.reset();
    
    unwantedEventsMap_.clear();
  }
  
  
  void RunMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
  {
    infoSpaceItems.push_back(std::make_pair("runNumber", &runNumber_));
    infoSpaceItems.push_back(std::make_pair("dataEvents", &dataEvents_));
    infoSpaceItems.push_back(std::make_pair("errorEvents", &errorEvents_));
    infoSpaceItems.push_back(std::make_pair("unwantedEvents", &unwantedEvents_));
  }
  
  
  void RunMonitorCollection::do_updateInfoSpaceItems()
  {
    MonitoredQuantity::Stats runNumberStats;
    runNumbersSeen_.getStats(runNumberStats);
    runNumber_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(runNumberStats.getLastSampleValue()));
    
    MonitoredQuantity::Stats eventIDsReceivedStats;
    eventIDsReceived_.getStats(eventIDsReceivedStats);
    dataEvents_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(eventIDsReceivedStats.getSampleCount()));
    
    MonitoredQuantity::Stats errorEventIDsReceivedStats;
    errorEventIDsReceived_.getStats(errorEventIDsReceivedStats);
    errorEvents_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(errorEventIDsReceivedStats.getSampleCount()));
    
    MonitoredQuantity::Stats unwantedEventStats;
    unwantedEventIDsReceived_.getStats(unwantedEventStats);
    unwantedEvents_ = static_cast<xdata::UnsignedInteger32>(
      static_cast<unsigned int>(unwantedEventStats.getSampleCount()));
  }
  
  
  void RunMonitorCollection::addUnwantedEvent(const I2OChain& ioc)
  {
    if ( ! alarmParams_.careAboutUnwantedEvents_ ) return;
    if ( ioc.faulty() || !ioc.complete() ) return;
    
    unwantedEventIDsReceived_.addSample(ioc.eventNumber());
    
    uint32_t outputModuleId = ioc.outputModuleId();
    
    boost::mutex::scoped_lock sl(unwantedEventMapLock_);
    
    UnwantedEventsMap::iterator pos = unwantedEventsMap_.lower_bound(outputModuleId);
    
    if(pos != unwantedEventsMap_.end() &&
      !(unwantedEventsMap_.key_comp()(outputModuleId, pos->first)))
    {
      // key already exists
      ++(pos->second.count);
    }
    else
    {
      UnwantedEvent newEvent(ioc);
      unwantedEventsMap_.insert(pos, UnwantedEventsMap::value_type(outputModuleId, newEvent));
    }
  }
  
  
  void RunMonitorCollection::checkForBadEvents()
  {
    alarmErrorEvents();
    
    boost::mutex::scoped_lock sl(unwantedEventMapLock_);
    std::for_each(unwantedEventsMap_.begin(), unwantedEventsMap_.end(),
      boost::bind(&RunMonitorCollection::alarmUnwantedEvents, this, _1));
  }
  
  
  void RunMonitorCollection::alarmErrorEvents()
  {
    if ( ! alarmParams_.isProductionSystem_ ) return;
    
    const std::string alarmName("ErrorEvents");
    
    MonitoredQuantity::Stats stats;
    errorEventIDsReceived_.getStats(stats);
    long long count = stats.getSampleCount(MonitoredQuantity::RECENT);
    
    if ( count >= alarmParams_.errorEvents_ )
    {
      std::ostringstream msg;
      msg << "Received " << count << " error events in the last "
        << stats.getDuration(MonitoredQuantity::RECENT).total_seconds() << "s.";
      XCEPT_DECLARE( stor::exception::ErrorEvents, xcept, msg.str() );
      sharedResources_->alarmHandler_->raiseAlarm( alarmName, AlarmHandler::ERROR, xcept );
    }
    else
    {
      sharedResources_->alarmHandler_->revokeAlarm( alarmName );
    }
  }
  
  
  void RunMonitorCollection::alarmUnwantedEvents(UnwantedEventsMap::value_type& val)
  {
    if ( ! alarmParams_.isProductionSystem_ ) return;
    
    if ( (val.second.count - val.second.previousCount) > alarmParams_.unwantedEvents_ )
    {
      std::ostringstream msg;
      msg << "Received " << val.second.count << " events"
        << " not tagged for any stream or consumer."
        << " Output module " << 
        sharedResources_->initMsgCollection_->getOutputModuleName(val.first)
        << " (id " << val.first << ")"
        << " HLT trigger bits: ";
      
      // This code snipped taken from evm:EventSelector::acceptEvent
      int byteIndex = 0;
      int subIndex  = 0;
      for (unsigned int pathIndex = 0;
           pathIndex < val.second.hltTriggerCount;
           ++pathIndex)
      {
        int state = val.second.bitList[byteIndex] >> (subIndex * 2);
        state &= 0x3;
        msg << state << " ";
        ++subIndex;
        if (subIndex == 4)
        { ++byteIndex;
          subIndex = 0;
        }
      }
      
      XCEPT_DECLARE( stor::exception::UnwantedEvents, xcept, msg.str() );
      sharedResources_->alarmHandler_->raiseAlarm( val.second.alarmName, AlarmHandler::ERROR, xcept );
      
      val.second.previousCount = val.second.count;
    }
    else if (val.second.count == val.second.previousCount)
      // no more unwanted events arrived
    {
      sharedResources_->alarmHandler_->revokeAlarm( val.second.alarmName );
    }
  }
  
  
  RunMonitorCollection::UnwantedEvent::UnwantedEvent(const I2OChain& ioc)
  : count(1), previousCount(0)
  {
    std::ostringstream str;
    str << "UnwantedEvent_" << nextId++;
    alarmName = str.str();
    hltTriggerCount = ioc.hltTriggerCount();
    ioc.hltTriggerBits(bitList);
  }
  
  uint32_t RunMonitorCollection::UnwantedEvent::nextId(0);
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
