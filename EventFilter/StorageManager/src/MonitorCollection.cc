// $Id: MonitorCollection.cc,v 1.11 2011/03/07 15:31:32 mommsen Exp $
/// @file: MonitorCollection.cc

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Exception.h"


namespace stor {
  
  MonitorCollection::MonitorCollection(const utils::Duration_t& updateInterval) :
  updateInterval_(updateInterval),
  lastCalculateStatistics_(boost::posix_time::not_a_date_time),
  infoSpaceUpdateNeeded_(false)
  {}
  
  
  void MonitorCollection::appendInfoSpaceItems(InfoSpaceItems& items)
  {
    // do any operations that are common for all child classes
    
    do_appendInfoSpaceItems(items);
  }
  
  
  void MonitorCollection::calculateStatistics(const utils::TimePoint_t& now)
  {
    if ( lastCalculateStatistics_.is_not_a_date_time() ||
      lastCalculateStatistics_ + updateInterval_ < now )
    {
      lastCalculateStatistics_ = now;
      do_calculateStatistics();
      infoSpaceUpdateNeeded_ = true;
    }
  }
  
  
  void MonitorCollection::updateInfoSpaceItems()
  {
    if (infoSpaceUpdateNeeded_)
    {
      do_updateInfoSpaceItems();
      infoSpaceUpdateNeeded_ = false;
    }
  }
  
  
  void MonitorCollection::reset(const utils::TimePoint_t& now)
  {
    do_reset();
    
    // Assure that the first update happens early.
    // This is important for long update intervals.
    lastCalculateStatistics_ = now - updateInterval_ + boost::posix_time::seconds(1);
    infoSpaceUpdateNeeded_ = true;
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
