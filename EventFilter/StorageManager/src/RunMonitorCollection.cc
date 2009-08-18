// $Id: RunMonitorCollection.cc,v 1.4 2009/07/20 13:07:28 mommsen Exp $
/// @file: RunMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"

using namespace stor;

RunMonitorCollection::RunMonitorCollection(const utils::duration_t& updateInterval) :
MonitorCollection(updateInterval),
_eventIDsReceived(updateInterval, 1),
_errorEventIDsReceived(updateInterval, 1),
_runNumbersSeen(updateInterval, 1),
_lumiSectionsSeen(updateInterval, 1)
{}


void RunMonitorCollection::do_calculateStatistics()
{
  _eventIDsReceived.calculateStatistics();
  _errorEventIDsReceived.calculateStatistics();
  _runNumbersSeen.calculateStatistics();
  _lumiSectionsSeen.calculateStatistics();
}


void RunMonitorCollection::do_reset()
{
  _eventIDsReceived.reset();
  _errorEventIDsReceived.reset();
  _runNumbersSeen.reset();
  _lumiSectionsSeen.reset();
}


void RunMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("runNumber", &_runNumber));

  // These infospace items were defined in the old SM
  // infoSpaceItems.push_back(std::make_pair("receivedEvents", &_receivedEvents));
  // infoSpaceItems.push_back(std::make_pair("receivedErrorEvents", &_receivedErrorEvents));
}


void RunMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats stats;
  
  _runNumbersSeen.getStats(stats);
  _runNumber = static_cast<xdata::UnsignedInteger32>(static_cast<unsigned int>(stats.getLastSampleValue()));
  
  // _eventIDsReceived.getStats(stats);
  // _receivedEvents = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
  
  // _errorEventIDsReceived.getStats(stats);
  // _receivedErrorEvents = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
}




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
