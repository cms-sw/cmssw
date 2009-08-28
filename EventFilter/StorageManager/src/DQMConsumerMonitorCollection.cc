// $Id: DQMConsumerMonitorCollection.cc,v 1.1 2009/08/24 14:31:50 mommsen Exp $
/// @file: DQMConsumerMonitorCollection.cc

#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

using namespace stor;


DQMConsumerMonitorCollection::DQMConsumerMonitorCollection(const utils::duration_t& updateInterval):
ConsumerMonitorCollection(updateInterval)
{}


void DQMConsumerMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("dqmConsumers", &_dqmConsumers));
}


void DQMConsumerMonitorCollection::do_updateInfoSpaceItems()
{
  boost::mutex::scoped_lock l( _mutex );
  _dqmConsumers = _smap.size();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
