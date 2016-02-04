// $Id: StatisticsReporter.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: StatisticsReporter.h 

#ifndef EventFilter_SMProxyServer_StatisticsReporter_h
#define EventFilter_SMProxyServer_StatisticsReporter_h

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/DataRetrieverMonitorCollection.h"
#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include <string>
#include <list>
#include <vector>
#include <utility>


namespace smproxy {

  /**
   * Singleton to keep track of all monitoring and statistics issues
   *
   * This class also starts the monitoring workloop to update the 
   * statistics for all MonitorCollections.
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:41:54 $
   */
  
  class StatisticsReporter : public toolbox::lang::Class, public xdata::ActionListener
  {
  public:
    
    explicit StatisticsReporter
    (
      xdaq::Application*,
      const QueueConfigurationParams&
    );
    
    virtual ~StatisticsReporter();

    const DataRetrieverMonitorCollection& getDataRetrieverMonitorCollection() const
    { return dataRetrieverMonCollection_; }

    DataRetrieverMonitorCollection& getDataRetrieverMonitorCollection()
    { return dataRetrieverMonCollection_; }

    const stor::DQMEventMonitorCollection& getDQMEventMonitorCollection() const
    { return dqmEventMonCollection_; }

    stor::DQMEventMonitorCollection& getDQMEventMonitorCollection()
    { return dqmEventMonCollection_; }


    const stor::EventConsumerMonitorCollection& getEventConsumerMonitorCollection() const
    { return eventConsumerMonCollection_; }

    stor::EventConsumerMonitorCollection& getEventConsumerMonitorCollection()
    { return eventConsumerMonCollection_; }


    const stor::DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection() const
    { return dqmConsumerMonCollection_; }

    stor::DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection()
    { return dqmConsumerMonCollection_; }


    /**
     * Create and start the monitoring workloop
     */
    void startWorkLoop(std::string workloopName);

    /**
     * Reset all monitored quantities
     */
    void reset();

    /**
     * Access alarm handler
     */
    stor::AlarmHandlerPtr alarmHandler() { return alarmHandler_; }

    /**
     * Update the variables put into the application info space
     */
    virtual void actionPerformed(xdata::Event&);


  private:

    typedef std::list<std::string> InfoSpaceItemNames;

    //Prevent copying of the StatisticsReporter
    StatisticsReporter(StatisticsReporter const&);
    StatisticsReporter& operator=(StatisticsReporter const&);

    void createMonitoringInfoSpace();
    void collectInfoSpaceItems();
    void putItemsIntoInfoSpace(stor::MonitorCollection::InfoSpaceItems&);
    bool monitorAction(toolbox::task::WorkLoop*);
    void calculateStatistics();
    void updateInfoSpace();

    xdaq::Application* app_;
    stor::AlarmHandlerPtr alarmHandler_;
    stor::utils::Duration_t monitoringSleepSec_;
    stor::utils::TimePoint_t lastMonitorAction_;

    DataRetrieverMonitorCollection dataRetrieverMonCollection_;
    stor::DQMEventMonitorCollection dqmEventMonCollection_;
    stor::EventConsumerMonitorCollection eventConsumerMonCollection_;
    stor::DQMConsumerMonitorCollection dqmConsumerMonCollection_;
    toolbox::task::WorkLoop* monitorWL_;      
    bool doMonitoring_;

    // Stuff dealing with the monitoring info space
    xdata::InfoSpace *infoSpace_;
    InfoSpaceItemNames infoSpaceItemNames_;

  };

  typedef boost::shared_ptr<StatisticsReporter> StatisticsReporterPtr;
  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_StatisticsReporter_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
