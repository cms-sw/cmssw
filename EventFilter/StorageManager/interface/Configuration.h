// $Id: Configuration.h,v 1.32 2012/06/08 10:20:33 mommsen Exp $
/// @file: Configuration.h 


#ifndef EventFilter_StorageManager_Configuration_h
#define EventFilter_StorageManager_Configuration_h

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "xdata/InfoSpace.h"
#include "xdata/String.h"
#include "xdata/Integer.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/Vector.h"

#include "boost/date_time/time_duration.hpp"
#include "boost/thread/mutex.hpp"


namespace stor
{
  /**
   * Data structure to hold configuration parameters
   * that are relevant for writing data to disk.
   */
  struct DiskWritingParams
  {
    std::string streamConfiguration_;
    std::string fileName_;
    std::string filePath_;
    std::string dbFilePath_;
    std::string setupLabel_;
    int nLogicalDisk_;
    int maxFileSizeMB_;
    double highWaterMark_;
    double failHighWaterMark_;
    utils::Duration_t lumiSectionTimeOut_;
    utils::Duration_t fileClosingTestInterval_;
    double fileSizeTolerance_;
    std::string faultyEventsStream_;
    bool checkAdler32_;

    typedef std::vector<std::string> OtherDiskPaths;
    OtherDiskPaths otherDiskPaths_;

    // not mapped to infospace params
    std::string smInstanceString_;
    std::string hostName_;
    int initialSafetyLevel_;  // what is this used for?
  };

  /**
   * Data structure to hold configuration parameters
   * that are relevant for the processing of DQM histograms.
   */
  struct DQMProcessingParams
  {
    bool collateDQM_;
    utils::Duration_t readyTimeDQM_;
    bool useCompressionDQM_;
    int compressionLevelDQM_;
    unsigned int discardDQMUpdatesForOlderLS_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are relevant for serving events to consumers.
   */
  struct EventServingParams
  {
    utils::Duration_t activeConsumerTimeout_;  // seconds
    int consumerQueueSize_;
    std::string consumerQueuePolicy_;
    utils::Duration_t _DQMactiveConsumerTimeout;  // seconds
    int _DQMconsumerQueueSize;
    std::string _DQMconsumerQueuePolicy;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used for the various queues in the system.
   */
  struct QueueConfigurationParams
  {
    unsigned int commandQueueSize_;
    unsigned int dqmEventQueueSize_;
    unsigned int dqmEventQueueMemoryLimitMB_;
    unsigned int fragmentQueueSize_;
    unsigned int fragmentQueueMemoryLimitMB_;
    unsigned int registrationQueueSize_;
    unsigned int streamQueueSize_;
    unsigned int streamQueueMemoryLimitMB_;
    unsigned int fragmentStoreMemoryLimitMB_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used by the various worker threads in the system.
   */
  struct WorkerThreadParams
  {
    boost::posix_time::time_duration FPdeqWaitTime_;
    boost::posix_time::time_duration DWdeqWaitTime_;
    boost::posix_time::time_duration DQMEPdeqWaitTime_;
    utils::Duration_t staleFragmentTimeOut_;
    utils::Duration_t monitoringSleepSec_;
    unsigned int throuphputAveragingCycles_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used by the resource monitor
   */
  struct ResourceMonitorParams
  {
    std::string sataUser_;    // user name to log into SATA controller

    struct WorkerParams
    {
      std::string user_;    // user name under which the workers are started
      std::string command_; // command name to grep for number of workers
      int expectedCount_;   // expected number of workers running on the node
    };
    WorkerParams injectWorkers_;
    WorkerParams copyWorkers_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used to enable sentinel alarms
   */
  struct AlarmParams
  {
    bool isProductionSystem_;      // indicates if the SM is running in production system.
                                   // If set to false, certain checks/alarms are disabled.
    bool careAboutUnwantedEvents_; // If set to false, the SM does not keep track of events
                                   // not served to any consumer or stream
    unsigned int errorEvents_;     // number of recent error events triggering an alarm
    unsigned int unwantedEvents_;  // number of unwanted events triggering an alarm

    AlarmParams() :
    isProductionSystem_(false),    // Initialize default to false here. This struct is
    careAboutUnwantedEvents_(true) // used before the actual values are set during the
    {};                            // configure transition.
  };

  /**
   * Free function to parse a storage manager configuration string
   * into the appropriate "configuration info" objects.
   */
  void parseStreamConfiguration(std::string cfgString,
                                EvtStrConfigListPtr evtCfgList,
                                ErrStrConfigListPtr errCfgList);

  /**
   * Class for managing configuration information from the infospace
   * and providing local copies of that information that are updated
   * only at requested times.
   *
   * $Author: mommsen $
   * $Revision: 1.32 $
   * $Date: 2012/06/08 10:20:33 $
   */

  class Configuration : public xdata::ActionListener
  {
  public:

    /**
     * Constructs a Configuration instance for the specified infospace
     * and application instance number.
     */
    Configuration(xdata::InfoSpace* infoSpace, unsigned long instanceNumber);

    /**
     * Destructor.
     */
    virtual ~Configuration()
    {
      // should we detach from the infospace???
    }

    /**
     * Returns a copy of the disk writing parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method) or
     * the most recent update of only the disk writing parameters
     * (see the updateDiskWritingParams() method).
     */
    struct DiskWritingParams getDiskWritingParams() const;

    /**
     * Returns a copy of the DQM processing parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct DQMProcessingParams getDQMProcessingParams() const;

    /**
     * Returns a copy of the event serving parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct EventServingParams getEventServingParams() const;

    /**
     * Returns a copy of the queue configuration parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct QueueConfigurationParams getQueueConfigurationParams() const;

    /**
     * Returns a copy of the worker thread parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct WorkerThreadParams getWorkerThreadParams() const;

    /**
     * Returns a copy of the resouce monitor parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct ResourceMonitorParams getResourceMonitorParams() const;

    /**
     * Returns a copy of the alarm parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct AlarmParams getAlarmParams() const;

    /**
     * Updates the local copy of all configuration parameters from
     * the infospace.
     */
    void updateAllParams();

    /**
     * Updates the local copy of the disk writing configuration parameters
     * from the infospace.
     */
    void updateDiskWritingParams();

    /**
     * Updates the local copy of run-based configuration parameter
     * from the infospace.
     */
    void updateRunParams();

    /**
     * Tests whether the stream configuration string has changed in
     * the infospace.  Returns true if it has changed, false if not.
     */
    bool streamConfigurationHasChanged() const;

    /**
     * Gets invoked when a operation is performed on the infospace
     * that we are interested in knowing about.
     */
    virtual void actionPerformed(xdata::Event& isEvt);

    /**
     * Sets the current list of event stream configuration info
     * objects.
     */
    void setCurrentEventStreamConfig(EvtStrConfigListPtr cfgList);

    /**
     * Sets the current list of error stream configuration info
     * objects.
     */
    void setCurrentErrorStreamConfig(ErrStrConfigListPtr cfgList);

    /**
     * Retrieves the current list of event stream configuration info
     * objects.
     */
    EvtStrConfigListPtr getCurrentEventStreamConfig() const;

    /**
     * Retrieves the current list of error stream configuration info
     * objects.
     */
    ErrStrConfigListPtr getCurrentErrorStreamConfig() const;

    /**
     * Get run number:
     */
    unsigned int getRunNumber() const;

  private:

    void setDiskWritingDefaults(unsigned long instanceNumber);
    void setDQMProcessingDefaults();
    void setEventServingDefaults();
    void setQueueConfigurationDefaults();
    void setWorkerThreadDefaults();
    void setResourceMonitorDefaults();
    void setAlarmDefaults();

    void setupDiskWritingInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupDQMProcessingInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupEventServingInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupQueueConfigurationInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupWorkerThreadInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupResourceMonitorInfoSpaceParams(xdata::InfoSpace* infoSpace);
    void setupAlarmInfoSpaceParams(xdata::InfoSpace* infoSpace);

    void updateLocalDiskWritingData();
    void updateLocalDQMProcessingData();
    void updateLocalEventServingData();
    void updateLocalQueueConfigurationData();
    void updateLocalWorkerThreadData();
    void updateLocalResourceMonitorData();
    void updateLocalAlarmData();
    void updateLocalRunNumberData();

    struct DiskWritingParams diskWriteParamCopy_;
    struct DQMProcessingParams dqmParamCopy_;
    struct EventServingParams eventServeParamCopy_;
    struct QueueConfigurationParams queueConfigParamCopy_;
    struct WorkerThreadParams workerThreadParamCopy_;
    struct ResourceMonitorParams resourceMonitorParamCopy_;
    struct AlarmParams alarmParamCopy_;

    mutable boost::mutex generalMutex_;

    std::string previousStreamCfg_;
    bool streamConfigurationChanged_;

    xdata::UnsignedInteger32 infospaceRunNumber_;
    unsigned int localRunNumber_;

    xdata::String streamConfiguration_;
    xdata::String fileName_;
    xdata::String filePath_;
    xdata::String dbFilePath_;
    xdata::Vector<xdata::String> otherDiskPaths_;
    xdata::String setupLabel_;
    xdata::Integer nLogicalDisk_;
    xdata::Integer maxFileSize_;
    xdata::Double highWaterMark_;
    xdata::Double failHighWaterMark_;
    xdata::Double lumiSectionTimeOut_;  // seconds
    xdata::Integer fileClosingTestInterval_;  // seconds
    xdata::Double fileSizeTolerance_;
    xdata::String faultyEventsStream_;
    xdata::Boolean checkAdler32_;

    xdata::Integer activeConsumerTimeout_;  // seconds
    xdata::Integer consumerQueueSize_;
    xdata::String  consumerQueuePolicy_;
    xdata::Integer _DQMactiveConsumerTimeout;  // seconds
    xdata::Integer _DQMconsumerQueueSize;
    xdata::String  _DQMconsumerQueuePolicy;

    xdata::Boolean collateDQM_;
    xdata::Integer readyTimeDQM_;  // seconds
    xdata::Boolean useCompressionDQM_;
    xdata::Integer compressionLevelDQM_;
    xdata::UnsignedInteger32 discardDQMUpdatesForOlderLS_;

    xdata::UnsignedInteger32 commandQueueSize_;
    xdata::UnsignedInteger32 dqmEventQueueSize_;
    xdata::UnsignedInteger32 dqmEventQueueMemoryLimitMB_;
    xdata::UnsignedInteger32 fragmentQueueSize_;
    xdata::UnsignedInteger32 fragmentQueueMemoryLimitMB_;
    xdata::UnsignedInteger32 registrationQueueSize_;
    xdata::UnsignedInteger32 streamQueueSize_;
    xdata::UnsignedInteger32 streamQueueMemoryLimitMB_;
    xdata::UnsignedInteger32 fragmentStoreMemoryLimitMB_;

    xdata::Double FPdeqWaitTime_;  // seconds
    xdata::Double DWdeqWaitTime_;  // seconds
    xdata::Double DQMEPdeqWaitTime_;  // seconds
    xdata::Double staleFragmentTimeOut_;  // seconds
    xdata::Double monitoringSleepSec_;  // seconds
    xdata::UnsignedInteger32 throuphputAveragingCycles_;

    xdata::String sataUser_;
    xdata::String injectWorkersUser_;
    xdata::String injectWorkersCommand_;
    xdata::Integer nInjectWorkers_;
    xdata::String copyWorkersUser_;
    xdata::String copyWorkersCommand_;
    xdata::Integer nCopyWorkers_;

    xdata::Boolean isProductionSystem_;
    xdata::Boolean careAboutUnwantedEvents_;
    xdata::UnsignedInteger32 errorEvents_;
    xdata::UnsignedInteger32 unwantedEvents_;

    mutable boost::mutex evtStrCfgMutex_;
    mutable boost::mutex errStrCfgMutex_;

    EvtStrConfigListPtr currentEventStreamConfig_;
    ErrStrConfigListPtr currentErrorStreamConfig_;
  };

  typedef boost::shared_ptr<Configuration> ConfigurationPtr;

} // namespace stor

#endif // EventFilter_StorageManager_Configuration_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

