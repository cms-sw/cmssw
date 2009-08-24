// $Id: RunMonitorCollection.cc,v 1.5 2009/08/18 08:55:12 mommsen Exp $
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
}


void RunMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats stats;
  
  _runNumbersSeen.getStats(stats);
  _runNumber = static_cast<xdata::UnsignedInteger32>(static_cast<unsigned int>(stats.getLastSampleValue()));
}




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
