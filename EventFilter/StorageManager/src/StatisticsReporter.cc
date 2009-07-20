// $Id: StatisticsReporter.cc,v 1.4 2009/07/09 15:34:29 mommsen Exp $
/// @file: StatisticsReporter.cc

#include <sstream>

#include "toolbox/net/URN.h"
#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdata/Event.h"
#include "xdata/InfoSpaceFactory.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;


StatisticsReporter::StatisticsReporter(xdaq::Application *app) :
_app(app),
_runMonCollection(),
_fragMonCollection(),
_filesMonCollection(),
_streamsMonCollection(),
_dataSenderMonCollection(),
_dqmEventMonCollection(),
_resourceMonCollection(app),
_stateMachineMonCollection(),
_throughputMonCollection(),
_monitorWL(0),
_doMonitoring(true)
{
  _eventConsumerMonitorCollection.reset( new ConsumerMonitorCollection() );
  _dqmConsumerMonitorCollection.reset( new ConsumerMonitorCollection() );

  createMonitoringInfoSpace();
  collectInfoSpaceItems();
  addRunInfoQuantitiesToApplicationInfoSpace();
}


void StatisticsReporter::startWorkLoop(std::string workloopName)
{
  try
  {
    std::string identifier = utils::getIdentifier(_app->getApplicationDescriptor());

    _monitorWL=
      toolbox::task::getWorkLoopFactory()->getWorkLoop(
        identifier + workloopName, "waiting");

    if ( ! _monitorWL->isActive() )
    {
      toolbox::task::ActionSignature* monitorAction = 
        toolbox::task::bind(this, &StatisticsReporter::monitorAction, 
          identifier + "MonitorAction");
      _monitorWL->submit(monitorAction);

      _monitorWL->activate();
    }
  }
  catch (xcept::Exception& e)
  {
    std::string msg = "Failed to start workloop 'StatisticsReporter' with 'MonitorAction'.";
    XCEPT_RETHROW(stor::exception::Monitoring, msg, e);
  }
}


StatisticsReporter::~StatisticsReporter()
{
  // Stop the monitoring activity
  _doMonitoring = false;

  // Cancel the workloop (will wait until the action has finished)
  if ( _monitorWL && _monitorWL->isActive() ) _monitorWL->cancel();
}



void StatisticsReporter::createMonitoringInfoSpace()
{
  // Create an infospace which can be monitored.
  // The naming follows the old SM scheme.
  // In future, the instance number should be included.
    
  std::ostringstream oss;
  oss << "urn:xdaq-monitorable-" << _app->getApplicationDescriptor()->getClassName();
  
  std::string errorMsg =
    "Failed to create monitoring info space " + oss.str();
  
  try
  {
    toolbox::net::URN urn = _app->createQualifiedInfoSpace(oss.str());
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


void StatisticsReporter::collectInfoSpaceItems()
{
  MonitorCollection::InfoSpaceItems infoSpaceItems;
  _infoSpaceItemNames.clear();

  _runMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _fragMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _filesMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _streamsMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _dataSenderMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _dqmEventMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _resourceMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _stateMachineMonCollection.appendInfoSpaceItems(infoSpaceItems);
  _eventConsumerMonitorCollection->appendInfoSpaceItems(infoSpaceItems);
  _dqmConsumerMonitorCollection->appendInfoSpaceItems(infoSpaceItems);
  _throughputMonCollection.appendInfoSpaceItems(infoSpaceItems);

  putItemsIntoInfoSpace(infoSpaceItems);
}


void StatisticsReporter::putItemsIntoInfoSpace(MonitorCollection::InfoSpaceItems& items)
{
  
  for ( MonitorCollection::InfoSpaceItems::const_iterator it = items.begin(),
          itEnd = items.end();
        it != itEnd;
        ++it )
  {
    try
    {
      // fireItemAvailable locks the infospace internally
      _infoSpace->fireItemAvailable(it->first, it->second);
    }
    catch(xdata::exception::Exception &e)
    {
      std::stringstream oss;
      
      oss << "Failed to put " << it->first;
      oss << " into info space " << _infoSpace->name();
      
      XCEPT_RETHROW(stor::exception::Monitoring, oss.str(), e);
    }

    // keep a list of info space names for the fireItemGroupChanged
    _infoSpaceItemNames.push_back(it->first);
  }
}


void StatisticsReporter::addRunInfoQuantitiesToApplicationInfoSpace()
{
  xdata::InfoSpace *infoSpace = _app->getApplicationInfoSpace();

  // bind the local xdata variables to the infospace
  infoSpace->fireItemAvailable("storedEvents", &_storedEvents);
  infoSpace->fireItemAvailable("closedFiles", &_closedFiles);

  // spacial handling for the monitoring values requested by the HLTSFM
  // we want to assure that the values are current when they are queried
  infoSpace->addItemRetrieveListener("closedFiles", this);
  infoSpace->addItemRetrieveListener("storedEvents", this);
}


bool StatisticsReporter::monitorAction(toolbox::task::WorkLoop* wl)
{
  utils::sleep(MonitoredQuantity::ExpectedCalculationInterval());

  std::string errorMsg = "Failed to update the monitoring information";

  try
  {
    calculateStatistics();
    updateInfoSpace();
  }
  catch(xcept::Exception &e)
  {
    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg << xcept::stdformat_exception_history(e));

    XCEPT_DECLARE_NESTED(stor::exception::Monitoring,
      sentinelException, errorMsg, e);
    _app->notifyQualified("error", sentinelException);
  }
  catch(std::exception &e)
  {
    errorMsg += ": ";
    errorMsg += e.what();

    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::Monitoring,
      sentinelException, errorMsg);
    _app->notifyQualified("error", sentinelException);
  }
  catch(...)
  {
    errorMsg += ": Unknown exception";

    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg);
    
    XCEPT_DECLARE(stor::exception::Monitoring,
      sentinelException, errorMsg);
    _app->notifyQualified("error", sentinelException);
  }

  return _doMonitoring;
}


void StatisticsReporter::calculateStatistics()
{
  _runMonCollection.calculateStatistics();
  _fragMonCollection.calculateStatistics();
  _filesMonCollection.calculateStatistics();
  _streamsMonCollection.calculateStatistics();
  _dataSenderMonCollection.calculateStatistics();
  _dqmEventMonCollection.calculateStatistics();
  _resourceMonCollection.calculateStatistics();
  _stateMachineMonCollection.calculateStatistics();
  _eventConsumerMonitorCollection->calculateStatistics();
  _dqmConsumerMonitorCollection->calculateStatistics();
  _throughputMonCollection.calculateStatistics();
}


void StatisticsReporter::updateInfoSpace()
{
  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();
    
    _runMonCollection.updateInfoSpaceItems();
    _fragMonCollection.updateInfoSpaceItems();
    _filesMonCollection.updateInfoSpaceItems();
    _streamsMonCollection.updateInfoSpaceItems();
    _dataSenderMonCollection.updateInfoSpaceItems();
    _dqmEventMonCollection.updateInfoSpaceItems();
    _resourceMonCollection.updateInfoSpaceItems();
    _stateMachineMonCollection.updateInfoSpaceItems();
    _eventConsumerMonitorCollection->updateInfoSpaceItems();
    _dqmConsumerMonitorCollection->updateInfoSpaceItems();
    _throughputMonCollection.updateInfoSpaceItems();

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


void StatisticsReporter::reset()
{
  // do not reset the stateMachineMonCollection, as we want to
  // keep the state machine history
  _runMonCollection.reset();
  _fragMonCollection.reset();
  _filesMonCollection.reset();
  _streamsMonCollection.reset();
  _dataSenderMonCollection.reset();
  _dqmEventMonCollection.reset();
  _resourceMonCollection.reset();
  _eventConsumerMonitorCollection->reset();
  _dqmConsumerMonitorCollection->reset();
  _throughputMonCollection.reset();
}


void StatisticsReporter::actionPerformed(xdata::Event& ispaceEvent)
{
  if (ispaceEvent.type() == "ItemRetrieveEvent")
  {
    std::string item =
      dynamic_cast<xdata::ItemRetrieveEvent&>(ispaceEvent).itemName();
    if (item == "closedFiles")
    {
      _filesMonCollection.updateInfoSpaceItems();
      try
      {
        _closedFiles.setValue( *(_infoSpace->find("closedFiles")) );
      }
      catch(xdata::exception::Exception& e)
      {
        _closedFiles = 0;
      }
    }
    else if (item == "storedEvents")
    {
      _streamsMonCollection.updateInfoSpaceItems();
      try
      {
        _storedEvents.setValue( *(_infoSpace->find("storedEvents")) );
      }
      catch(xdata::exception::Exception& e)
      {
        _storedEvents = 0;
      }
    } 
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
