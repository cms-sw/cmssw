// $Id: StatisticsReporter.h,v 1.15 2011/11/08 10:48:40 mommsen Exp $
/// @file: StatisticsReporter.h 

#ifndef EventFilter_StorageManager_StatisticsReporter_h
#define EventFilter_StorageManager_StatisticsReporter_h

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"
#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachineMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include <string>
#include <list>
#include <vector>
#include <utility>


namespace stor {

  /**
   * Singleton to keep track of all monitoring and statistics issues
   *
   * This class also starts the monitoring workloop to update the 
   * statistics for all MonitorCollections.
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2011/11/08 10:48:40 $
   */
  
  class StatisticsReporter : public toolbox::lang::Class, public xdata::ActionListener
  {
  public:
    
    explicit StatisticsReporter
    (
      xdaq::Application*,
      SharedResourcesPtr
    );
    
    virtual ~StatisticsReporter();

    const RunMonitorCollection& getRunMonitorCollection() const
    { return runMonCollection_; }

    RunMonitorCollection& getRunMonitorCollection()
    { return runMonCollection_; }


    const FragmentMonitorCollection& getFragmentMonitorCollection() const
    { return fragMonCollection_; }

    FragmentMonitorCollection& getFragmentMonitorCollection()
    { return fragMonCollection_; }


    const FilesMonitorCollection& getFilesMonitorCollection() const
    { return filesMonCollection_; }

    FilesMonitorCollection& getFilesMonitorCollection()
    { return filesMonCollection_; }


    const StreamsMonitorCollection& getStreamsMonitorCollection() const
    { return streamsMonCollection_; }

    StreamsMonitorCollection& getStreamsMonitorCollection()
    { return streamsMonCollection_; }


    const DataSenderMonitorCollection& getDataSenderMonitorCollection() const
    { return dataSenderMonCollection_; }

    DataSenderMonitorCollection& getDataSenderMonitorCollection()
    { return dataSenderMonCollection_; }


    const DQMEventMonitorCollection& getDQMEventMonitorCollection() const
    { return dqmEventMonCollection_; }

    DQMEventMonitorCollection& getDQMEventMonitorCollection()
    { return dqmEventMonCollection_; }


    const ResourceMonitorCollection& getResourceMonitorCollection() const
    { return resourceMonCollection_; }

    ResourceMonitorCollection& getResourceMonitorCollection()
    { return resourceMonCollection_; }


    const StateMachineMonitorCollection& getStateMachineMonitorCollection() const
    { return stateMachineMonCollection_; }

    StateMachineMonitorCollection& getStateMachineMonitorCollection()
    { return stateMachineMonCollection_; }


    const EventConsumerMonitorCollection& getEventConsumerMonitorCollection() const
    { return eventConsumerMonCollection_; }

    EventConsumerMonitorCollection& getEventConsumerMonitorCollection()
    { return eventConsumerMonCollection_; }


    const DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection() const
    { return dqmConsumerMonCollection_; }

    DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection()
    { return dqmConsumerMonCollection_; }


    const ThroughputMonitorCollection& getThroughputMonitorCollection() const
    { return throughputMonCollection_; }

    ThroughputMonitorCollection& getThroughputMonitorCollection()
    { return throughputMonCollection_; }


    /**
     * Create and start the monitoring workloop
     */
    void startWorkLoop(std::string workloopName);

    /**
     * Reset all monitored quantities
     */
    void reset();

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
    void putItemsIntoInfoSpace(MonitorCollection::InfoSpaceItems&);
    void addRunInfoQuantitiesToApplicationInfoSpace();
    bool monitorAction(toolbox::task::WorkLoop*);
    void calculateStatistics();
    void updateInfoSpace();

    xdaq::Application* app_;
    utils::Duration_t monitoringSleepSec_;
    utils::TimePoint_t lastMonitorAction_;

    RunMonitorCollection runMonCollection_;
    FragmentMonitorCollection fragMonCollection_;
    FilesMonitorCollection filesMonCollection_;
    StreamsMonitorCollection streamsMonCollection_;
    DataSenderMonitorCollection dataSenderMonCollection_;
    DQMEventMonitorCollection dqmEventMonCollection_;
    ResourceMonitorCollection resourceMonCollection_;
    StateMachineMonitorCollection stateMachineMonCollection_;
    EventConsumerMonitorCollection eventConsumerMonCollection_;
    DQMConsumerMonitorCollection dqmConsumerMonCollection_;
    ThroughputMonitorCollection throughputMonCollection_;
    toolbox::task::WorkLoop* monitorWL_;      
    bool doMonitoring_;

    // Stuff dealing with the monitoring info space
    xdata::InfoSpace *infoSpace_;
    InfoSpaceItemNames infoSpaceItemNames_;

    // These values have to be in the application infospace as
    // the HLTSFM queries them from the application infospace at 
    // the end of the run to put them into the RunInfo DB. The HTLSFM
    // uses the application infospace, and not the monitoring infospace.
    xdata::String stateName_;
    xdata::UnsignedInteger32 storedEvents_;
    xdata::UnsignedInteger32 closedFiles_;

  };

  typedef boost::shared_ptr<StatisticsReporter> StatisticsReporterPtr;
  
} // namespace stor

#endif // EventFilter_StorageManager_StatisticsReporter_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
