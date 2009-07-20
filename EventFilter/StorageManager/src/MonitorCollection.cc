// $Id: MonitorCollection.cc,v 1.3 2009/07/09 15:34:28 mommsen Exp $
/// @file: MonitorCollection.cc

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace stor;


void MonitorCollection::calculateStatistics()
{
  // do any operations that are common for all child classes

  do_calculateStatistics();
}


void MonitorCollection::appendInfoSpaceItems(InfoSpaceItems& items)
{
  // do any operations that are common for all child classes

  do_appendInfoSpaceItems(items);
}


void MonitorCollection::updateInfoSpaceItems()
{
  // do any operations that are common for all child classes

  do_updateInfoSpaceItems();
}


void MonitorCollection::reset()
{
  // do any operations that are common for all child classes

  do_reset();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
