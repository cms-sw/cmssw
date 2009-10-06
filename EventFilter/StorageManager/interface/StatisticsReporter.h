// $Id: StatisticsReporter.h,v 1.10 2009/09/22 14:54:50 dshpakov Exp $
/// @file: StatisticsReporter.h 

#ifndef StorageManager_StatisticsReporter_h
#define StorageManager_StatisticsReporter_h

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

  class AlarmHandler;

  /**
   * Singleton to keep track of all monitoring and statistics issues
   *
   * This class also starts the monitoring workloop to update the 
   * statistics for all MonitorCollections.
   *
   * $Author: dshpakov $
   * $Revision: 1.10 $
   * $Date: 2009/09/22 14:54:50 $
   */
  
  class StatisticsReporter : public toolbox::lang::Class, public xdata::ActionListener
  {
  public:
    
    explicit StatisticsReporter
    (
      xdaq::Application*,
      const utils::duration_t& monitoringSleepSec
    );
    
    virtual ~StatisticsReporter();

    const RunMonitorCollection& getRunMonitorCollection() const
    { return _runMonCollection; }

    RunMonitorCollection& getRunMonitorCollection()
    { return _runMonCollection; }


    const FragmentMonitorCollection& getFragmentMonitorCollection() const
    { return _fragMonCollection; }

    FragmentMonitorCollection& getFragmentMonitorCollection()
    { return _fragMonCollection; }


    const FilesMonitorCollection& getFilesMonitorCollection() const
    { return _filesMonCollection; }

    FilesMonitorCollection& getFilesMonitorCollection()
    { return _filesMonCollection; }


    const StreamsMonitorCollection& getStreamsMonitorCollection() const
    { return _streamsMonCollection; }

    StreamsMonitorCollection& getStreamsMonitorCollection()
    { return _streamsMonCollection; }


    const DataSenderMonitorCollection& getDataSenderMonitorCollection() const
    { return _dataSenderMonCollection; }

    DataSenderMonitorCollection& getDataSenderMonitorCollection()
    { return _dataSenderMonCollection; }


    const DQMEventMonitorCollection& getDQMEventMonitorCollection() const
    { return _dqmEventMonCollection; }

    DQMEventMonitorCollection& getDQMEventMonitorCollection()
    { return _dqmEventMonCollection; }


    const ResourceMonitorCollection& getResourceMonitorCollection() const
    { return _resourceMonCollection; }

    ResourceMonitorCollection& getResourceMonitorCollection()
    { return _resourceMonCollection; }


    const StateMachineMonitorCollection& getStateMachineMonitorCollection() const
    { return _stateMachineMonCollection; }

    StateMachineMonitorCollection& getStateMachineMonitorCollection()
    { return _stateMachineMonCollection; }


    const EventConsumerMonitorCollection& getEventConsumerMonitorCollection() const
    { return _eventConsumerMonCollection; }

    EventConsumerMonitorCollection& getEventConsumerMonitorCollection()
    { return _eventConsumerMonCollection; }


    const DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection() const
    { return _dqmConsumerMonCollection; }

    DQMConsumerMonitorCollection& getDQMConsumerMonitorCollection()
    { return _dqmConsumerMonCollection; }


    const ThroughputMonitorCollection& getThroughputMonitorCollection() const
    { return _throughputMonCollection; }

    ThroughputMonitorCollection& getThroughputMonitorCollection()
    { return _throughputMonCollection; }


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
    typedef boost::shared_ptr<AlarmHandler> AlarmHandlerPtr;
    AlarmHandlerPtr alarmHandler() { return _alarmHandler; }

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

    xdaq::Application* _app;
    AlarmHandlerPtr _alarmHandler;
    utils::duration_t _monitoringSleepSec;
    utils::time_point_t _lastMonitorAction;

    RunMonitorCollection _runMonCollection;
    FragmentMonitorCollection _fragMonCollection;
    FilesMonitorCollection _filesMonCollection;
    StreamsMonitorCollection _streamsMonCollection;
    DataSenderMonitorCollection _dataSenderMonCollection;
    DQMEventMonitorCollection _dqmEventMonCollection;
    ResourceMonitorCollection _resourceMonCollection;
    StateMachineMonitorCollection _stateMachineMonCollection;
    EventConsumerMonitorCollection _eventConsumerMonCollection;
    DQMConsumerMonitorCollection _dqmConsumerMonCollection;
    ThroughputMonitorCollection _throughputMonCollection;
    toolbox::task::WorkLoop* _monitorWL;      
    bool _doMonitoring;

    // Stuff dealing with the monitoring info space
    xdata::InfoSpace *_infoSpace;
    InfoSpaceItemNames _infoSpaceItemNames;

    // These values have to be in the application infospace as
    // the HLTSFM queries them from the application infospace at 
    // the end of the run to put them into the RunInfo DB. The HTLSFM
    // uses the application infospace, and not the monitoring infospace.
    xdata::String _stateName;
    xdata::UnsignedInteger32 _storedEvents;
    xdata::UnsignedInteger32 _closedFiles;

  };

  typedef boost::shared_ptr<StatisticsReporter> StatisticsReporterPtr;
  
} // namespace stor

#endif // StorageManager_StatisticsReporter_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
