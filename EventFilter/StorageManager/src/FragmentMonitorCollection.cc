// $Id: FragmentMonitorCollection.cc,v 1.9 2011/03/07 15:31:32 mommsen Exp $
/// @file: FragmentMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"


namespace stor {
  
  FragmentMonitorCollection::FragmentMonitorCollection(const utils::Duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  allFragmentSizes_(updateInterval, boost::posix_time::seconds(5)),
  allFragmentBandwidth_(updateInterval, boost::posix_time::seconds(5)),
  eventFragmentSizes_(updateInterval, boost::posix_time::seconds(5)),
  eventFragmentBandwidth_(updateInterval, boost::posix_time::seconds(5)),
  dqmEventFragmentSizes_(updateInterval, boost::posix_time::seconds(300)),
  dqmEventFragmentBandwidth_(updateInterval, boost::posix_time::seconds(300))
  {}
  
  
  void FragmentMonitorCollection::addFragmentSample(const double bytecount)
  {
    double mbytes = bytecount / 0x100000;
    allFragmentSizes_.addSample(mbytes);
  }
  
  
  void FragmentMonitorCollection::addEventFragmentSample(const double bytecount)
  {
    double mbytes = bytecount / 0x100000;
    allFragmentSizes_.addSample(mbytes);
    eventFragmentSizes_.addSample(mbytes);
  }
  
  
  void FragmentMonitorCollection::addDQMEventFragmentSample(const double bytecount)
  {
    double mbytes = bytecount / 0x100000;
    allFragmentSizes_.addSample(mbytes);
    dqmEventFragmentSizes_.addSample(mbytes);
  }
  
  
  void FragmentMonitorCollection::getStats(FragmentStats& stats) const
  {
    getAllFragmentSizeMQ().getStats(stats.allFragmentSizeStats);
    getEventFragmentSizeMQ().getStats(stats.eventFragmentSizeStats);
    getDQMEventFragmentSizeMQ().getStats(stats.dqmEventFragmentSizeStats);
    
    getAllFragmentBandwidthMQ().getStats(stats.allFragmentBandwidthStats);
    getEventFragmentBandwidthMQ().getStats(stats.eventFragmentBandwidthStats);
    getDQMEventFragmentBandwidthMQ().getStats(stats.dqmEventFragmentBandwidthStats);
  }
  
  
  void FragmentMonitorCollection::do_calculateStatistics()
  {
    allFragmentSizes_.calculateStatistics();
    eventFragmentSizes_.calculateStatistics();
    dqmEventFragmentSizes_.calculateStatistics();
    
    MonitoredQuantity::Stats stats;
    allFragmentSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      allFragmentBandwidth_.addSample(stats.getLastValueRate());
    }
    allFragmentBandwidth_.calculateStatistics();
    
    eventFragmentSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      eventFragmentBandwidth_.addSample(stats.getLastValueRate());
    }
    eventFragmentBandwidth_.calculateStatistics();
    
    dqmEventFragmentSizes_.getStats(stats);
    if (stats.getSampleCount() > 0) {
      dqmEventFragmentBandwidth_.addSample(stats.getLastValueRate());
    }
    dqmEventFragmentBandwidth_.calculateStatistics();
  }
  
  
  void FragmentMonitorCollection::do_reset()
  {
    allFragmentSizes_.reset();
    eventFragmentSizes_.reset();
    dqmEventFragmentSizes_.reset();
    
    allFragmentBandwidth_.reset();
    eventFragmentBandwidth_.reset();
    dqmEventFragmentBandwidth_.reset();
  }
  
  
  void FragmentMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
  {
    infoSpaceItems.push_back(std::make_pair("receivedFrames", &receivedFrames_));
    infoSpaceItems.push_back(std::make_pair("instantBandwidth", &instantBandwidth_));
    infoSpaceItems.push_back(std::make_pair("instantRate", &instantRate_));
  }
  
  
  void FragmentMonitorCollection::do_updateInfoSpaceItems()
  {
    MonitoredQuantity::Stats stats;
    
    allFragmentSizes_.getStats(stats);
    receivedFrames_ = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
    instantRate_           = static_cast<xdata::Double>(stats.getSampleRate(MonitoredQuantity::RECENT));
    
    allFragmentBandwidth_.getStats(stats);
    instantBandwidth_ = static_cast<xdata::Double>(stats.getValueRate(MonitoredQuantity::RECENT));
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
