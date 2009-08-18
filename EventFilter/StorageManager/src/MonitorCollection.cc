// $Id: MonitorCollection.cc,v 1.4 2009/07/20 13:07:27 mommsen Exp $
/// @file: MonitorCollection.cc

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace stor;


MonitorCollection::MonitorCollection(const utils::duration_t& updateInterval) :
_updateInterval(updateInterval)
{}


void MonitorCollection::appendInfoSpaceItems(InfoSpaceItems& items)
{
  // do any operations that are common for all child classes

  do_appendInfoSpaceItems(items);
}


void MonitorCollection::calculateStatistics(const utils::time_point_t& now)
{
  if (_lastCalculateStatistics + _updateInterval < now)
  {
    _lastCalculateStatistics = now;
    do_calculateStatistics();
    _infoSpaceUpdateNeeded = true;
  }
}


void MonitorCollection::updateInfoSpaceItems()
{
  if (_infoSpaceUpdateNeeded)
  {
    do_updateInfoSpaceItems();
    _infoSpaceUpdateNeeded = false;
  }
}


void MonitorCollection::reset(const utils::time_point_t& now)
{
  _lastCalculateStatistics = now;
  do_reset();
  _infoSpaceUpdateNeeded = true;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
