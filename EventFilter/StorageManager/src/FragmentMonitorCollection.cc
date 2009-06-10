// $Id$

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"

using namespace stor;

FragmentMonitorCollection::FragmentMonitorCollection(xdaq::Application *app) :
MonitorCollection(app)
{
  _allFragmentSizes.setNewTimeWindowForRecentResults(5);
  _allFragmentBandwidth.setNewTimeWindowForRecentResults(5);
  _eventFragmentSizes.setNewTimeWindowForRecentResults(5);
  _eventFragmentBandwidth.setNewTimeWindowForRecentResults(5);
  _dqmEventFragmentSizes.setNewTimeWindowForRecentResults(300);
  _dqmEventFragmentBandwidth.setNewTimeWindowForRecentResults(300);

  _infoSpaceItems.push_back(std::make_pair("receivedFrames", &_receivedFrames));
  _infoSpaceItems.push_back(std::make_pair("instantBandwidth", &_instantBandwidth));
  _infoSpaceItems.push_back(std::make_pair("instantRate", &_instantRate));

  // There infospace items were defined in the old SM
  // _infoSpaceItems.push_back(std::make_pair("duration", &_duration));
  // _infoSpaceItems.push_back(std::make_pair("totalSamples", &_totalSamples));
  // _infoSpaceItems.push_back(std::make_pair("dqmRecords", &_dqmRecords));
  // _infoSpaceItems.push_back(std::make_pair("meanBandwidth", &_meanBandwidth));
  // _infoSpaceItems.push_back(std::make_pair("meanLatency", &_meanLatency));
  // _infoSpaceItems.push_back(std::make_pair("meanRate", &_meanRate));
  // _infoSpaceItems.push_back(std::make_pair("receivedVolume", &_receivedVolume));
  // _infoSpaceItems.push_back(std::make_pair("receivedPeriod4Stats", &_receivedPeriod4Stats));
  // _infoSpaceItems.push_back(std::make_pair("receivedSamples4Stats", &_receivedSamples4Stats));
  // _infoSpaceItems.push_back(std::make_pair("instantLatency", &_instantLatency));
  // _infoSpaceItems.push_back(std::make_pair("maxBandwidth", &_maxBandwidth));
  // _infoSpaceItems.push_back(std::make_pair("minBandwidth", &_minBandwidth));
  // _infoSpaceItems.push_back(std::make_pair("receivedDQMPeriod4Stats", &_receivedDQMPeriod4Stats));
  // _infoSpaceItems.push_back(std::make_pair("receivedDQMSamples4Stats", &_receivedDQMSamples4Stats));

  putItemsIntoInfoSpace();
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


void FragmentMonitorCollection::do_updateInfoSpace()
{
  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();
    MonitoredQuantity::Stats stats;

    _allFragmentSizes.getStats(stats);
    // _duration       = static_cast<xdata::Double>(stats.getDuration());
    _receivedFrames = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
    // _meanRate       = static_cast<xdata::Double>(stats.getSampleRate());
    // _meanLatency    = static_cast<xdata::Double>(stats.getSampleLatency());
    // _receivedVolume = static_cast<xdata::Double>(stats.getValueSum());
    // _receivedPeriod4Stats  = static_cast<xdata::UnsignedInteger32>(static_cast<unsigned int> (stats.getDuration(MonitoredQuantity::RECENT)));
    // _receivedSamples4Stats = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount(MonitoredQuantity::RECENT));
    _instantRate           = static_cast<xdata::Double>(stats.getSampleRate(MonitoredQuantity::RECENT));
    // _instantLatency        = static_cast<xdata::Double>(stats.getSampleLatency(MonitoredQuantity::RECENT));


    // _totalSamples   = _receivedFrames;
    // _dqmEventFragmentSizes.getStats(stats);
    // _dqmRecords               = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());
    // _receivedDQMPeriod4Stats  = static_cast<xdata::UnsignedInteger32>(static_cast<unsigned int>(stats.getDuration(MonitoredQuantity::RECENT)));
    // _receivedDQMSamples4Stats = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount(MonitoredQuantity::RECENT));

    
    _allFragmentBandwidth.getStats(stats);
    // _meanBandwidth    = static_cast<xdata::Double>(stats.getValueRate());
    _instantBandwidth = static_cast<xdata::Double>(stats.getValueRate(MonitoredQuantity::RECENT));
    // _maxBandwidth     = static_cast<xdata::Double>(stats.getValueMax(MonitoredQuantity::RECENT));
    // _minBandwidth     = static_cast<xdata::Double>(stats.getValueMin(MonitoredQuantity::RECENT));

    _infoSpace->unlock();
  }
  catch(std::exception &e)
  {
    _infoSpace->unlock();
 
    errorMsg += ": ";
    errorMsg += e.what();
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }
  catch (...)
  {
    _infoSpace->unlock();
 
    errorMsg += " : unknown exception";
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }

  try
  {
    // The fireItemGroupChanged locks the infospace
    _infoSpace->fireItemGroupChanged(_infoSpaceItemNames, this);
  }
  catch (xdata::exception::Exception &e)
  {
    XCEPT_RETHROW(stor::exception::Monitoring, errorMsg, e);
  }
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




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
