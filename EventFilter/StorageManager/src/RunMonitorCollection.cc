// $Id$

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"

using namespace stor;

RunMonitorCollection::RunMonitorCollection(xdaq::Application *app) :
MonitorCollection(app)
{
  _infoSpaceItems.push_back(std::make_pair("runNumber", &_runNumber));

  // These infospace items were defined in the old SM
  // _infoSpaceItems.push_back(std::make_pair("receivedEvents", &_receivedEvents));
  // _infoSpaceItems.push_back(std::make_pair("receivedErrorEvents", &_receivedErrorEvents));

  putItemsIntoInfoSpace();
}


void RunMonitorCollection::do_calculateStatistics()
{
  _eventIDsReceived.calculateStatistics();
  _errorEventIDsReceived.calculateStatistics();
  _runNumbersSeen.calculateStatistics();
  _lumiSectionsSeen.calculateStatistics();
}


void RunMonitorCollection::do_updateInfoSpace()
{
  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();
    MonitoredQuantity::Stats stats;

    _runNumbersSeen.getStats(stats);
    _runNumber = static_cast<xdata::UnsignedInteger32>(static_cast<unsigned int>(stats.getLastSampleValue()));

    // _eventIDsReceived.getStats(stats);
    // _receivedEvents = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());

    // _errorEventIDsReceived.getStats(stats);
    // _receivedErrorEvents = static_cast<xdata::UnsignedInteger32>(stats.getSampleCount());

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
    XCEPT_RETHROW(stor::exception::Infospace, errorMsg, e);
  }
}


void RunMonitorCollection::do_reset()
{
  _eventIDsReceived.reset();
  _errorEventIDsReceived.reset();
  _runNumbersSeen.reset();
  _lumiSectionsSeen.reset();
}




/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
