// $Id$

#include <sstream>

#include "toolbox/net/URN.h"
#include "xdata/InfoSpaceFactory.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Exception.h"

using namespace stor;

xdata::InfoSpace* MonitorCollection::_infoSpace = 0;

MonitorCollection::MonitorCollection(xdaq::Application *app)
{
  if (!_infoSpace)
  {
    // Create an infospace which can be monitored.
    // The naming follows the old SM scheme.
    // In future, the instance number should be included.
    
    std::ostringstream oss;
    oss << "urn:xdaq-monitorable-" << app->getApplicationDescriptor()->getClassName();
    
    std::string errorMsg =
      "Failed to create monitoring info space " + oss.str();
    
    try
    {
      toolbox::net::URN urn = app->createQualifiedInfoSpace(oss.str());
      xdata::getInfoSpaceFactory()->lock();
      _infoSpace = xdata::getInfoSpaceFactory()->get(urn.toString());
      xdata::getInfoSpaceFactory()->unlock();
    }
    catch(xdata::exception::Exception &e)
    {
      xdata::getInfoSpaceFactory()->unlock();
      
      XCEPT_RETHROW(stor::exception::Infospace, errorMsg, e);
    }
    catch (...)
    {
      xdata::getInfoSpaceFactory()->unlock();
      
      errorMsg += " : unknown exception";
      XCEPT_RAISE(stor::exception::Infospace, errorMsg);
    }
  }
}

void MonitorCollection::update()
{
  calculateStatistics();
  updateInfoSpace();
}


void MonitorCollection::calculateStatistics()
{
  // do any operations that are common for all child classes

  do_calculateStatistics();
}


void MonitorCollection::updateInfoSpace()
{
  // do any operations that are common for all child classes

  do_updateInfoSpace();
}


void MonitorCollection::reset()
{
  // do any operations that are common for all child classes

  do_reset();
}


void MonitorCollection::putItemsIntoInfoSpace()
{
  MonitorCollection::infoSpaceItems_t::const_iterator itor;

  for (
    itor=_infoSpaceItems.begin(); 
    itor!=_infoSpaceItems.end();
    ++itor)
  {
    try
    {
      // fireItemAvailable locks the infospace internally
      _infoSpace->fireItemAvailable(itor->first, itor->second);
    }
    catch(xdata::exception::Exception &e)
    {
      std::stringstream oss;
      
      oss << "Failed to put " << itor->first;
      oss << " into info space " << _infoSpace->name();
      
      XCEPT_RETHROW(stor::exception::Monitoring, oss.str(), e);
    }

    // keep a list of info space names for the fireItemGroupChanged
    _infoSpaceItemNames.push_back(itor->first);
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
