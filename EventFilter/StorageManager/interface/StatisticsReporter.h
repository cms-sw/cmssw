// $Id$

#ifndef StorageManager_StatisticsReporter_h
#define StorageManager_StatisticsReporter_h

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StateMachineMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/ThroughputMonitorCollection.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include <string>

namespace stor {

  /**
   * Singleton to keep track of all monitoring and statistics issues
   *
   * This class also starts the monitoring workloop to update the 
   * statistics for all MonitorCollections.
   *
   * $Author$
   * $Revision$
   * $Date$
   */
  
  class StatisticsReporter : public toolbox::lang::Class, public xdata::ActionListener
  {
  public:
    
    explicit StatisticsReporter(xdaq::Application*);
    
    virtual ~StatisticsReporter();

    typedef boost::shared_ptr<ConsumerMonitorCollection> CMCPtr;

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


    CMCPtr getEventConsumerMonitorCollection()
    {
      return _eventConsumerMonitorCollection;
    }

    CMCPtr getDQMConsumerMonitorCollection()
    {
      return _dqmConsumerMonitorCollection;
    }

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
     * Update the variables put into the application info space
     */
    virtual void actionPerformed(xdata::Event&);


  private:

    //Prevent copying of the StatisticsReporter
    StatisticsReporter(StatisticsReporter const&);
    StatisticsReporter& operator=(StatisticsReporter const&);

    bool monitorAction(toolbox::task::WorkLoop*);
    void addRunInfoQuantitiesToApplicationInfoSpace();

    xdaq::Application* _app;
    RunMonitorCollection _runMonCollection;
    FragmentMonitorCollection _fragMonCollection;
    FilesMonitorCollection _filesMonCollection;
    StreamsMonitorCollection _streamsMonCollection;
    DataSenderMonitorCollection _dataSenderMonCollection;
    DQMEventMonitorCollection _dqmEventMonCollection;
    ResourceMonitorCollection _resourceMonCollection;
    StateMachineMonitorCollection _stateMachineMonCollection;
    CMCPtr _eventConsumerMonitorCollection;
    CMCPtr _dqmConsumerMonitorCollection;
    ThroughputMonitorCollection _throughputMonCollection;
    toolbox::task::WorkLoop* _monitorWL;      
    bool _doMonitoring;

    // These values have to be in the application infospace as
    // the HLTSFM queries them from the application infospace at 
    // the end of the run to put them into the RunInfo DB. The HTLSFM
    // uses the application infospace, and not the monitoring infospace.
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
