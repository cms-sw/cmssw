// $Id: StatisticsReporter.cc,v 1.3 2011/05/09 11:03:34 mommsen Exp $
/// @file: StatisticsReporter.cc

#include <sstream>

#include "toolbox/net/URN.h"
#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoopFactory.h"
#include "xcept/tools.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xdata/Event.h"
#include "xdata/InfoSpaceFactory.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/SMProxyServer/interface/Exception.h"
#include "EventFilter/SMProxyServer/interface/StatisticsReporter.h"


namespace smproxy {
  
  StatisticsReporter::StatisticsReporter
  (
    xdaq::Application *app,
    const QueueConfigurationParams& qcp
  ) :
  app_(app),
  alarmHandler_(new stor::AlarmHandler(app)),
  monitoringSleepSec_(qcp.monitoringSleepSec_),
  dataRetrieverMonCollection_(monitoringSleepSec_, alarmHandler_),
  dqmEventMonCollection_(monitoringSleepSec_*5),
  eventConsumerMonCollection_(monitoringSleepSec_),
  dqmConsumerMonCollection_(monitoringSleepSec_),
  doMonitoring_(monitoringSleepSec_>boost::posix_time::seconds(0))
  {
    reset();
    createMonitoringInfoSpace();
    collectInfoSpaceItems();
  }
  
  
  void StatisticsReporter::startWorkLoop(std::string workloopName)
  {
    if ( !doMonitoring_ ) return;
    
    try
    {
      std::string identifier =
        stor::utils::getIdentifier(app_->getApplicationDescriptor());
      
      monitorWL_=
        toolbox::task::getWorkLoopFactory()->getWorkLoop(
          identifier + workloopName, "waiting");
      
      if ( ! monitorWL_->isActive() )
      {
        toolbox::task::ActionSignature* monitorAction = 
          toolbox::task::bind(this, &StatisticsReporter::monitorAction, 
            identifier + "MonitorAction");
        monitorWL_->submit(monitorAction);
        
        lastMonitorAction_ = stor::utils::getCurrentTime();
        monitorWL_->activate();
      }
    }
    catch (xcept::Exception& e)
    {
      std::string msg =
        "Failed to start workloop 'StatisticsReporter' with 'MonitorAction'.";
      XCEPT_RETHROW(exception::Monitoring, msg, e);
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
      
      XCEPT_RETHROW(exception::Infospace, errorMsg, e);
    }
    catch (...)
    {
      xdata::getInfoSpaceFactory()->unlock();
      
      errorMsg += " : unknown exception";
      XCEPT_RAISE(exception::Infospace, errorMsg);
    }
  }
  
  
  void StatisticsReporter::collectInfoSpaceItems()
  {
    stor::MonitorCollection::InfoSpaceItems infoSpaceItems;
    infoSpaceItemNames_.clear();
    
    dataRetrieverMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    dqmEventMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    eventConsumerMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    dqmConsumerMonCollection_.appendInfoSpaceItems(infoSpaceItems);
    
    putItemsIntoInfoSpace(infoSpaceItems);
  }
  
  
  void StatisticsReporter::putItemsIntoInfoSpace
  (
    stor::MonitorCollection::InfoSpaceItems& items
  )
  {
    
    for ( stor::MonitorCollection::InfoSpaceItems::const_iterator it = items.begin(),
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
        
        XCEPT_RETHROW(exception::Monitoring, oss.str(), e);
      }
      
      // keep a list of info space names for the fireItemGroupChanged
      infoSpaceItemNames_.push_back(it->first);
    }
  }
  
  
  bool StatisticsReporter::monitorAction(toolbox::task::WorkLoop* wl)
  {
    stor::utils::sleepUntil(lastMonitorAction_ + monitoringSleepSec_);
    lastMonitorAction_ = stor::utils::getCurrentTime();
    
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
      
      XCEPT_DECLARE_NESTED(exception::Monitoring,
        sentinelException, errorMsg, e);
      app_->notifyQualified("error", sentinelException);
    }
    catch(std::exception &e)
    {
      errorMsg += ": ";
      errorMsg += e.what();
      
      LOG4CPLUS_ERROR(app_->getApplicationLogger(),
        errorMsg);
      
      XCEPT_DECLARE(exception::Monitoring,
        sentinelException, errorMsg);
      app_->notifyQualified("error", sentinelException);
    }
    catch(...)
    {
      errorMsg += ": Unknown exception";
      
      LOG4CPLUS_ERROR(app_->getApplicationLogger(),
        errorMsg);
      
      XCEPT_DECLARE(exception::Monitoring,
        sentinelException, errorMsg);
      app_->notifyQualified("error", sentinelException);
    }
    
    return doMonitoring_;
  }
  
  
  void StatisticsReporter::calculateStatistics()
  {
    const stor::utils::TimePoint_t now = stor::utils::getCurrentTime();
    
    dataRetrieverMonCollection_.calculateStatistics(now);
    dqmEventMonCollection_.calculateStatistics(now);
    eventConsumerMonCollection_.calculateStatistics(now);
    dqmConsumerMonCollection_.calculateStatistics(now);
  }
  
  
  void StatisticsReporter::updateInfoSpace()
  {
    std::string errorMsg =
      "Failed to update values of items in info space " + infoSpace_->name();
    
    // Lock the infospace to assure that all items are consistent
    try
    {
      infoSpace_->lock();
      
      dataRetrieverMonCollection_.updateInfoSpaceItems();
      dqmEventMonCollection_.updateInfoSpaceItems();
      eventConsumerMonCollection_.updateInfoSpaceItems();
      dqmConsumerMonCollection_.updateInfoSpaceItems();
      
      infoSpace_->unlock();
    }
    catch(std::exception &e)
    {
      infoSpace_->unlock();
      
      errorMsg += ": ";
      errorMsg += e.what();
      XCEPT_RAISE(exception::Monitoring, errorMsg);
    }
    catch (...)
    {
      infoSpace_->unlock();
      
      errorMsg += " : unknown exception";
      XCEPT_RAISE(exception::Monitoring, errorMsg);
    }
    
    try
    {
      // The fireItemGroupChanged locks the infospace
      infoSpace_->fireItemGroupChanged(infoSpaceItemNames_, this);
    }
    catch (xdata::exception::Exception &e)
    {
      XCEPT_RETHROW(exception::Monitoring, errorMsg, e);
    }
  }
  
  
  void StatisticsReporter::reset()
  {
    const stor::utils::TimePoint_t now = stor::utils::getCurrentTime();
    
    dataRetrieverMonCollection_.reset(now);
    dqmEventMonCollection_.reset(now);
    eventConsumerMonCollection_.reset(now);
    dqmConsumerMonCollection_.reset(now);
    
    alarmHandler_->clearAllAlarms();
  }
  
  
  void StatisticsReporter::actionPerformed(xdata::Event& ispaceEvent)
  {}
  
} // namespace smproxy

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
