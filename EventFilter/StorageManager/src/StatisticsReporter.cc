// $Id: StatisticsReporter.cc,v 1.22 2011/11/08 10:48:41 mommsen Exp $
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
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {
  
  StatisticsReporter::StatisticsReporter
  (
    xdaq::Application *app,
    SharedResourcesPtr sr
  ) :
  app_(app),
  monitoringSleepSec_(sr->configuration_->
    getWorkerThreadParams().monitoringSleepSec_),
  runMonCollection_(monitoringSleepSec_, sr),
  fragMonCollection_(monitoringSleepSec_),
  filesMonCollection_(monitoringSleepSec_*5),
  streamsMonCollection_(monitoringSleepSec_),
  dataSenderMonCollection_(monitoringSleepSec_, sr->alarmHandler_),
  dqmEventMonCollection_(monitoringSleepSec_*5),
  resourceMonCollection_(monitoringSleepSec_*600, sr->alarmHandler_),
  stateMachineMonCollection_(monitoringSleepSec_),
  eventConsumerMonCollection_(monitoringSleepSec_),
  dqmConsumerMonCollection_(monitoringSleepSec_),
  throughputMonCollection_(monitoringSleepSec_,
    sr->configuration_->getWorkerThreadParams().throuphputAveragingCycles_),
  monitorWL_(0),
  doMonitoring_(monitoringSleepSec_>boost::posix_time::seconds(0))
  {
    reset();
    createMonitoringInfoSpace();
    collectInfoSpaceItems();
    addRunInfoQuantitiesToApplicationInfoSpace();
  }
  
  
  void StatisticsReporter::startWorkLoop(std::string workloopName)
  {
    if ( !doMonitoring_ ) return;
    
    try
    {
      std::string identifier = utils::getIdentifier(app_->getApplicationDescriptor());
      
      monitorWL_=
        toolbox::task::getWorkLoopFactory()->getWorkLoop(
          identifier + workloopName, "waiting");
      
      if ( ! monitorWL_->isActive() )
      {
        toolbox::task::ActionSignature* monitorAction = 
          toolbox::task::bind(this, &StatisticsReporter::monitorAction, 
            identifier + "MonitorAction");
        monitorWL_->submit(monitorAction);
        
        lastMonitorAction_ = utils::getCurrentTime();
        monitorWL_->activate();
      }
    }
    catch (xcept::Exception& e)
    {
      const std::string msg =
        "Failed to start workloop 'StatisticsReporter' with 'MonitorAction'.";
      XCEPT_RETHROW(stor::exception::Monitoring, msg, e);
    }
  }
  
  
  StatisticsReporter::~StatisticsReporter()
  {
    // Stop the monitoring activity
    doMonitoring_ = false;
    
    // Cancel the workloop (will wait until the action has finished)
    if ( monitorWL_ && monitorWL_->isActive() ) monitorWL_->cancel();
  }
  
  
  void StatisticsReporter::createMonitoringInfoSpace()
  {
    // Create an infospace which can be monitored.
    // The naming follows the old SM scheme.
    // In future, the instance number should be included.
    
    std::ostringstream oss;
    oss << "urn:xdaq-monitorable-" << app_->getApplicationDescriptor()->getClassName();
    
    std::string errorMsg =
      "Failed to create monitoring info space " + oss.str();
    
    try
    {
      toolbox::net::URN urn = app_->createQualifiedInfoSpace(oss.str());
      xdata::getInfoSpaceFactory()->lock();
      infoSpace_ = xdata::getInfoSpaceFactory()->get(urn.toString());
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
    infoSpaceItemNames_.clear();
    
    runMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    fragMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    filesMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    streamsMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    dataSenderMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    dqmEventMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    resourceMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    stateMachineMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    eventConsumerMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    dqmConsumerMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    throughputMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    
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
        infoSpace_->fireItemAvailable(it->first, it->second);
      }
      catch(xdata::exception::Exception &e)
      {
        std::stringstream oss;
        
        oss << "Failed to put " << it->first;
        oss << " into info space " << infoSpace_->name();
        
        XCEPT_RETHROW(stor::exception::Monitoring, oss.str(), e);
      }
      
      // keep a list of info space names for the fireItemGroupChanged
      infoSpaceItemNames_.push_back(it->first);
    }
  }
  
  
  void StatisticsReporter::addRunInfoQuantitiesToApplicationInfoSpace()
  {
    xdata::InfoSpace *infoSpace = app_->getApplicationInfoSpace();
    
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("stateName", &stateName_);
    infoSpace->fireItemAvailable("storedEvents", &storedEvents_);
    infoSpace->fireItemAvailable("closedFiles", &closedFiles_);
    
    // spacial handling for the monitoring values requested by the HLTSFM
    // we want to assure that the values are current when they are queried
    infoSpace->addItemRetrieveListener("stateName", this);
    infoSpace->addItemRetrieveListener("storedEvents", this);
    infoSpace->addItemRetrieveListener("closedFiles", this);
  }
  
  
  bool StatisticsReporter::monitorAction(toolbox::task::WorkLoop* wl)
  {
    utils::sleepUntil(lastMonitorAction_ + monitoringSleepSec_);
    lastMonitorAction_ = utils::getCurrentTime();
    
    std::string errorMsg = "Failed to update the monitoring information";
    
    try
    {
      calculateStatistics();
      updateInfoSpace();
    }
    catch(xcept::Exception &e)
    {
      LOG4CPLUS_ERROR(app_->getApplicationLogger(),
        errorMsg << xcept::stdformat_exception_history(e));
      
      XCEPT_DECLARE_NESTED(stor::exception::Monitoring,
        sentinelException, errorMsg, e);
      app_->notifyQualified("error", sentinelException);
    }
    catch(std::exception &e)
    {
      errorMsg += ": ";
      errorMsg += e.what();
      
      LOG4CPLUS_ERROR(app_->getApplicationLogger(),
        errorMsg);
      
      XCEPT_DECLARE(stor::exception::Monitoring,
        sentinelException, errorMsg);
      app_->notifyQualified("error", sentinelException);
    }
    catch(...)
    {
      errorMsg += ": Unknown exception";
      
      LOG4CPLUS_ERROR(app_->getApplicationLogger(),
        errorMsg);
      
      XCEPT_DECLARE(stor::exception::Monitoring,
        sentinelException, errorMsg);
      app_->notifyQualified("error", sentinelException);
    }
    
    return doMonitoring_;
  }
  
  
  void StatisticsReporter::calculateStatistics()
  {
    const utils::TimePoint_t now = utils::getCurrentTime();
    
    runMonCollection_.calculateStatistics(now);
    fragMonCollection_.calculateStatistics(now);
    filesMonCollection_.calculateStatistics(now);
    streamsMonCollection_.calculateStatistics(now);
    dataSenderMonCollection_.calculateStatistics(now);
    dqmEventMonCollection_.calculateStatistics(now);
    resourceMonCollection_.calculateStatistics(now);
    stateMachineMonCollection_.calculateStatistics(now);
    eventConsumerMonCollection_.calculateStatistics(now);
    dqmConsumerMonCollection_.calculateStatistics(now);
    throughputMonCollection_.calculateStatistics(now);
  }
  
  
  void StatisticsReporter::updateInfoSpace()
  {
    std::string errorMsg =
      "Failed to update values of items in info space " + infoSpace_->name();
    
    // Lock the infospace to assure that all items are consistent
    try
    {
      infoSpace_->lock();
      
      runMonCollection_.updateInfoSpaceItems();
      fragMonCollection_.updateInfoSpaceItems();
      filesMonCollection_.updateInfoSpaceItems();
      streamsMonCollection_.updateInfoSpaceItems();
      dataSenderMonCollection_.updateInfoSpaceItems();
      dqmEventMonCollection_.updateInfoSpaceItems();
      resourceMonCollection_.updateInfoSpaceItems();
      stateMachineMonCollection_.updateInfoSpaceItems();
      eventConsumerMonCollection_.updateInfoSpaceItems();
      dqmConsumerMonCollection_.updateInfoSpaceItems();
      throughputMonCollection_.updateInfoSpaceItems();
      
      infoSpace_->unlock();
    }
    catch(std::exception &e)
    {
      infoSpace_->unlock();
      
      errorMsg += ": ";
      errorMsg += e.what();
      XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
    }
    catch (...)
    {
      infoSpace_->unlock();
      
      errorMsg += " : unknown exception";
      XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
    }
    
    try
    {
      // The fireItemGroupChanged locks the infospace
      infoSpace_->fireItemGroupChanged(infoSpaceItemNames_, this);
    }
    catch (xdata::exception::Exception &e)
    {
      XCEPT_RETHROW(stor::exception::Monitoring, errorMsg, e);
    }
  }
  
  
  void StatisticsReporter::reset()
  {
    const utils::TimePoint_t now = utils::getCurrentTime();
    
    // do not reset the stateMachineMonCollection, as we want to
    // keep the state machine history
    runMonCollection_.reset(now);
    fragMonCollection_.reset(now);
    filesMonCollection_.reset(now);
    streamsMonCollection_.reset(now);
    dataSenderMonCollection_.reset(now);
    dqmEventMonCollection_.reset(now);
    resourceMonCollection_.reset(now);
    eventConsumerMonCollection_.reset(now);
    dqmConsumerMonCollection_.reset(now);
    throughputMonCollection_.reset(now);
  }
  
  
  void StatisticsReporter::actionPerformed(xdata::Event& ispaceEvent)
  {
    if (ispaceEvent.type() == "ItemRetrieveEvent")
    {
      std::string item =
        dynamic_cast<xdata::ItemRetrieveEvent&>(ispaceEvent).itemName();
      if (item == "closedFiles")
      {
        filesMonCollection_.updateInfoSpaceItems();
        try
        {
          closedFiles_.setValue( *(infoSpace_->find("closedFiles")) );
        }
        catch(xdata::exception::Exception& e)
        {
          closedFiles_ = 0;
        }
      }
      else if (item == "storedEvents")
      {
        streamsMonCollection_.updateInfoSpaceItems();
        try
        {
          storedEvents_.setValue( *(infoSpace_->find("storedEvents")) );
        }
        catch(xdata::exception::Exception& e)
        {
          storedEvents_ = 0;
        }
      } 
      else if (item == "stateName")
      {
        stateMachineMonCollection_.updateInfoSpaceItems();
        try
        {
          stateName_.setValue( *(infoSpace_->find("stateName")) );
        }
        catch(xdata::exception::Exception& e)
        {
          stateName_ = "unknown";
        }
      } 
    }
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
