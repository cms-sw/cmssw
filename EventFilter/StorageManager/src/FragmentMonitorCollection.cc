// $Id: FragmentMonitorCollection.cc,v 1.7 2009/12/08 15:03:52 mommsen Exp $
/// @file: FragmentMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"

using namespace stor;

FragmentMonitorCollection::FragmentMonitorCollection(const utils::duration_t& updateInterval) :
MonitorCollection(updateInterval),
_allFragmentSizes(updateInterval, boost::posix_time::seconds(5)),
_allFragmentBandwidth(updateInterval, boost::posix_time::seconds(5)),
_eventFragmentSizes(updateInterval, boost::posix_time::seconds(5)),
_eventFragmentBandwidth(updateInterval, boost::posix_time::seconds(5)),
_dqmEventFragmentSizes(updateInterval, boost::posix_time::seconds(300)),
_dqmEventFragmentBandwidth(updateInterval, boost::posix_time::seconds(300))
{}


void FragmentMonitorCollection::addFragmentSample(const double bytecount)
{
  double mbytes = bytecount / 0x100000;
  _allFragmentSizes.addSample(mbytes);
}


void FragmentMonitorCollection::addEventFragmentSample(const double bytecount)
{
  double mbytes = bytecount / 0x100000;
  _allFragmentSizes.addSample(mbytes);
  _eventFragmentSizes.addSample(mbytes);
}


void FragmentMonitorCollection::addDQMEventFragmentSample(const double bytecount)
{
  double mbytes = bytecount / 0x100000;
  _allFragmentSizes.addSample(mbytes);
  _dqmEventFragmentSizes.addSample(mbytes);
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
  _allFragmentSizes.calculateStatistics();
  _eventFragmentSizes.calculateStatistics();
  _dqmEventFragmentSizes.calculateStatistics();

  MonitoredQuantity::Stats stats;
  _allFragmentSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _allFragmentBandwidth.addSample(stats.getLastValueRate());
  }
  _allFragmentBandwidth.calculateStatistics();

  _eventFragmentSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _eventFragmentBandwidth.addSample(stats.getLastValueRate());
  }
  _eventFragmentBandwidth.calculateStatistics();

  _dqmEventFragmentSizes.getStats(stats);
  if (stats.getSampleCount() > 0) {
    _dqmEventFragmentBandwidth.addSample(stats.getLastValueRate());
  }
  _dqmEventFragmentBandwidth.calculateStatistics();
}


void FragmentMonitorCollection::do_reset()
{
  _allFragmentSizes.reset();
  _eventFragmentSizes.reset();
  _dqmEventFragmentSizes.reset();

  _allFragmentBandwidth.reset();
  _eventFragmentBandwidth.reset();
  _dqmEventFragmentBandwidth.reset();
}


void FragmentMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("receivedFrames", &_receivedFrames));
  infoSpaceItems.push_back(std::make_pair("instantBandwidth", &_instantBandwidth));
  infoSpaceItems.push_back(std::make_pair("instantRate", &_instantRate));
}


void FragmentMonitorCollection::do_updateInfoSpaceItems()
{
  MonitoredQuantity::Stats stats;
  
  _allFragmentSizes.getStats(stats);
  _receivedFrames = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
  _instantRate           = static_cast<xdata::Double>(stats.getSampleRate(MonitoredQuantity::RECENT));
  
  _allFragmentBandwidth.getStats(stats);
  _instantBandwidth = static_cast<xdata::Double>(stats.getValueRate(MonitoredQuantity::RECENT));
}




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
