// $Id: EventConsumerMonitorCollection.cc,v 1.2 2009/08/28 16:41:26 mommsen Exp $
/// @file: EventConsumerMonitorCollection.cc

#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

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
