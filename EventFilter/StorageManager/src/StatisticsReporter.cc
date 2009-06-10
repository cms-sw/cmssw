// $Id$

#include <string>
#include <sstream>

#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdata/Event.h"
#include "xdata/InfoSpace.h"

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;


StatisticsReporter::StatisticsReporter(xdaq::Application *app) :
_app(app),
_runMonCollection(app),
_fragMonCollection(app),
_filesMonCollection(app),
_streamsMonCollection(app),
_dataSenderMonCollection(app),
_dqmEventMonCollection(app),
_resourceMonCollection(app),
_stateMachineMonCollection(app),
_throughputMonCollection(app),
_monitorWL(0),
_doMonitoring(true)
{
  _eventConsumerMonitorCollection.reset( new ConsumerMonitorCollection( app ) );
  _dqmConsumerMonitorCollection.reset( new ConsumerMonitorCollection( app ) );

  addRunInfoQuantitiesToApplicationInfoSpace();
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


bool StatisticsReporter::monitorAction(toolbox::task::WorkLoop* wl)
{
  utils::sleep(MonitoredQuantity::ExpectedCalculationInterval());

  std::string errorMsg = "Failed to update the monitoring information";

  try
  {
    _runMonCollection.update();
    _fragMonCollection.update();
    _filesMonCollection.update();
    _streamsMonCollection.update();
    _dataSenderMonCollection.update();
    _dqmEventMonCollection.update();
    _resourceMonCollection.update();
    _stateMachineMonCollection.update();
    _eventConsumerMonitorCollection->update();
    _dqmConsumerMonitorCollection->update();
    _throughputMonCollection.update();
  }
  catch(xcept::Exception &e)
  {
    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg << xcept::stdformat_exception_history(e));

    #ifndef STOR_BYPASS_SENTINEL
    XCEPT_DECLARE_NESTED(stor::exception::Monitoring,
      sentinelException, errorMsg, e);
    _app->notifyQualified("error", sentinelException);
    #endif
  }
  catch(std::exception &e)
  {
    errorMsg += ": ";
    errorMsg += e.what();

    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg);
    
    #ifndef STOR_BYPASS_SENTINEL
    XCEPT_DECLARE(stor::exception::Monitoring,
      sentinelException, errorMsg);
    _app->notifyQualified("error", sentinelException);
    #endif
  }
  catch(...)
  {
    errorMsg += ": Unknown exception";

    LOG4CPLUS_ERROR(_app->getApplicationLogger(),
      errorMsg);
    
    #ifndef STOR_BYPASS_SENTINEL
    XCEPT_DECLARE(stor::exception::Monitoring,
      sentinelException, errorMsg);
    _app->notifyQualified("error", sentinelException);
    #endif
  }

  return _doMonitoring;
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
      _filesMonCollection.updateInfoSpace();
      xdata::InfoSpace* ispace = _filesMonCollection.getMonitoringInfoSpace();
      try
      {
        _closedFiles.setValue( *(ispace->find("closedFiles")) );
      }
      catch(xdata::exception::Exception& e)
      {
        _closedFiles = 0;
      }
    }
    else if (item == "storedEvents")
    {
      _streamsMonCollection.updateInfoSpace();
      xdata::InfoSpace* ispace = _streamsMonCollection.getMonitoringInfoSpace();
      try
      {
        _storedEvents.setValue( *(ispace->find("storedEvents")) );
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
