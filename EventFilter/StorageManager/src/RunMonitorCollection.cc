// $Id: RunMonitorCollection.cc,v 1.7 2010/02/09 14:54:04 mommsen Exp $
/// @file: RunMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <boost/bind.hpp>

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"

using namespace stor;

RunMonitorCollection::RunMonitorCollection
(
  const utils::duration_t& updateInterval,
  boost::shared_ptr<AlarmHandler> ah
) :
MonitorCollection(updateInterval),
_eventIDsReceived(updateInterval, 1),
_errorEventIDsReceived(updateInterval, 1),
_unwantedEventIDsReceived(updateInterval, 1),
_runNumbersSeen(updateInterval, 1),
_lumiSectionsSeen(updateInterval, 1),
_eolsSeen(updateInterval, 1),
_alarmHandler(ah)
{}


void RunMonitorCollection::configureAlarms(AlarmParams const& alarmParams)
{
  _alarmParams = alarmParams;
}


void RunMonitorCollection::do_calculateStatistics()
{
  _eventIDsReceived.calculateStatistics();
  _errorEventIDsReceived.calculateStatistics();
  _unwantedEventIDsReceived.calculateStatistics();
  _runNumbersSeen.calculateStatistics();
  _lumiSectionsSeen.calculateStatistics();
  _eolsSeen.calculateStatistics();

  checkForBadEvents();
}


void RunMonitorCollection::do_reset()
{
  _eventIDsReceived.reset();
  _errorEventIDsReceived.reset();
  _unwantedEventIDsReceived.reset();
  _runNumbersSeen.reset();
  _lumiSectionsSeen.reset();
  _eolsSeen.reset();

  _unwantedEventsMap.clear();
}


void RunMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("runNumber", &_runNumber));
  infoSpaceItems.push_back(std::make_pair("unwantedEvents", &_unwantedEvents));
}


void RunMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats runNumberStats;
  _runNumbersSeen.getStats(runNumberStats);
  _runNumber = static_cast<xdata::UnsignedInteger32>(
    static_cast<unsigned int>(runNumberStats.getLastSampleValue()));

  MonitoredQuantity::Stats unwantedEventStats;
  _unwantedEventIDsReceived.getStats(unwantedEventStats);
  _unwantedEvents = static_cast<xdata::UnsignedInteger32>(
    static_cast<unsigned int>(unwantedEventStats.getSampleCount()));
}


void RunMonitorCollection::addUnwantedEvent(const I2OChain& ioc)
{
  _unwantedEventIDsReceived.addSample(ioc.eventNumber());

  UnwantedEventKey key(ioc);

  boost::mutex::scoped_lock sl(_unwantedEventMapLock);

  UnwantedEventsMap::iterator pos = _unwantedEventsMap.lower_bound(key);

  if(pos != _unwantedEventsMap.end() && !(_unwantedEventsMap.key_comp()(key, pos->first)))
  {
    // key already exists
    ++(pos->second.count);
  }
  else
  {
    UnwantedEventValue newVal;
    _unwantedEventsMap.insert(pos, UnwantedEventsMap::value_type(key, newVal));
  }
}


void RunMonitorCollection::checkForBadEvents()
{
  alarmErrorEvents();

  boost::mutex::scoped_lock sl(_unwantedEventMapLock);
  std::for_each(_unwantedEventsMap.begin(), _unwantedEventsMap.end(),
    boost::bind(&RunMonitorCollection::alarmUnwantedEvents, this, _1));
}


void RunMonitorCollection::alarmErrorEvents()
{
  if ( ! _alarmParams._isProductionSystem ) return;

  const std::string alarmName("ErrorEvents");

  MonitoredQuantity::Stats stats;
  _errorEventIDsReceived.getStats(stats);
  long long count = stats.getSampleCount(MonitoredQuantity::RECENT);

  if ( count >= _alarmParams._errorEvents )
  {
    std::ostringstream msg;
    msg << "Received " << count << " error events in the last "
      << stats.getDuration(MonitoredQuantity::RECENT) << "s.";
    XCEPT_DECLARE( stor::exception::ErrorEvents, xcept, msg.str() );
    _alarmHandler->raiseAlarm( alarmName, AlarmHandler::ERROR, xcept );
  }
  else
  {
    _alarmHandler->revokeAlarm( alarmName );
  }
}


void RunMonitorCollection::alarmUnwantedEvents(UnwantedEventsMap::value_type& val)
{
  if ( ! _alarmParams._isProductionSystem ) return;

  if ( (val.second.previousCount == 0)
    || (val.second.count - val.second.previousCount > _alarmParams._unwantedEvents) )
  {
    std::ostringstream msg;
    msg << "Received " << val.second.count << " events"
      << " not tagged for any stream or consumer."
      << " Output module id " << val.first.outputModuleId
      << " HLT trigger bits: ";

    // This code snipped taken from evm:EventSelector::acceptEvent
    int byteIndex = 0;
    int subIndex  = 0;
    for (unsigned int pathIndex = 0; pathIndex < val.first.hltTriggerCount; ++pathIndex)
    {
      int state = val.first.bitList[byteIndex] >> (subIndex * 2);
      state &= 0x3;
      msg << state << " ";
      ++subIndex;
      if (subIndex == 4)
      { ++byteIndex;
        subIndex = 0;
      }
    }
    
    XCEPT_DECLARE( stor::exception::UnwantedEvents, xcept, msg.str() );
    _alarmHandler->raiseAlarm( val.second.alarmName, AlarmHandler::ERROR, xcept );

    val.second.previousCount = val.second.count;
  }
  else if (val.second.count == val.second.previousCount)
    // no more unwanted events arrived
  {
    _alarmHandler->revokeAlarm( val.second.alarmName );
  }
}


RunMonitorCollection::UnwantedEventKey::UnwantedEventKey(const I2OChain& ioc)
{
  outputModuleId = ioc.outputModuleId();
  hltTriggerCount = ioc.hltTriggerCount();
  ioc.hltTriggerBits(bitList);
}


bool RunMonitorCollection::UnwantedEventKey::operator<(UnwantedEventKey const& other) const
{
  if ( outputModuleId != other.outputModuleId ) return outputModuleId < other.outputModuleId;
  if ( hltTriggerCount != other.hltTriggerCount ) return hltTriggerCount < other.hltTriggerCount;

  for (unsigned int i = 0 ; i < bitList.size() ; ++i)
  {
    if ( bitList[i] != other.bitList[i] ) return bitList[i] < other.bitList[i];
  }
  return false;
}


RunMonitorCollection::UnwantedEventValue::UnwantedEventValue()
  : count(1), previousCount(0)
{
  std::ostringstream str;
  str << "UnwantedEvent_" << nextId++;
  alarmName = str.str();
}

uint32 RunMonitorCollection::UnwantedEventValue::nextId(0);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
