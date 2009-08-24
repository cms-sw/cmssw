// $Id: EventConsumerMonitorCollection.cc,v 1.5 2009/08/18 08:55:12 mommsen Exp $
/// @file: EventConsumerMonitorCollection.cc

#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"

using namespace stor;


EventConsumerMonitorCollection::EventConsumerMonitorCollection(const utils::duration_t& updateInterval):
ConsumerMonitorCollection(updateInterval)
{}


void EventConsumerMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("eventConsumers", &_eventConsumers));
}


void EventConsumerMonitorCollection::do_updateInfoSpaceItems()
{
  boost::mutex::scoped_lock l( _mutex );
  _eventConsumers = _smap.size();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
