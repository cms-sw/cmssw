// $Id: Configuration.h,v 1.23 2010/09/28 16:25:29 mommsen Exp $
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

#include "boost/thread/mutex.hpp"


namespace stor
{
  /**
   * Data structure to hold configuration parameters
   * that are relevant for writing data to disk.
   */
  struct DiskWritingParams
  {
    std::string _streamConfiguration;
    std::string _fileName;
    std::string _filePath;
    std::string _dbFilePath;
    std::string _fileCatalog;
    std::string _setupLabel;
    int _nLogicalDisk;
    int _maxFileSizeMB;
    double _highWaterMark;
    double _failHighWaterMark;
    utils::duration_t _lumiSectionTimeOut;
    utils::duration_t _fileClosingTestInterval;
    double _fileSizeTolerance;
    std::string _faultyEventsStream;

    typedef std::vector<std::string> OtherDiskPaths;
    OtherDiskPaths _otherDiskPaths;

    // not mapped to infospace params
    std::string _smInstanceString;
    std::string _hostName;
    int _initialSafetyLevel;  // what is this used for?
  };

  /**
   * Data structure to hold configuration parameters
   * that are relevant for the processing of DQM histograms.
   */
  struct DQMProcessingParams
  {
    bool _collateDQM;
    bool _archiveDQM;
    std::string _filePrefixDQM;
    utils::duration_t _archiveIntervalDQM;
    utils::duration_t _purgeTimeDQM;
    utils::duration_t _readyTimeDQM;
    bool _useCompressionDQM;
    int _compressionLevelDQM;
  };

  /**
   * Data structure to hold configuration parameters
   * that are relevant for serving events to consumers.
   */
  struct EventServingParams
  {
    utils::duration_t _activeConsumerTimeout;  // seconds
    int _consumerQueueSize;
    std::string _consumerQueuePolicy;
    utils::duration_t _DQMactiveConsumerTimeout;  // seconds
    int _DQMconsumerQueueSize;
    std::string _DQMconsumerQueuePolicy;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used for the various queues in the system.
   */
  struct QueueConfigurationParams
  {
    unsigned int _commandQueueSize;
    unsigned int _dqmEventQueueSize;
    unsigned int _dqmEventQueueMemoryLimitMB;
    unsigned int _fragmentQueueSize;
    unsigned int _fragmentQueueMemoryLimitMB;
    unsigned int _registrationQueueSize;
    unsigned int _streamQueueSize;
    unsigned int _streamQueueMemoryLimitMB;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used by the various worker threads in the system.
   */
  struct WorkerThreadParams
  {
    utils::duration_t _FPdeqWaitTime;
    utils::duration_t _DWdeqWaitTime;
    utils::duration_t _DQMEPdeqWaitTime;
    utils::duration_t _staleFragmentTimeOut;
    utils::duration_t _monitoringSleepSec;
    unsigned int _throuphputAveragingCycles;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used by the resource monitor
   */
  struct ResourceMonitorParams
  {
    std::string _sataUser;    // user name to log into SATA controller

    struct WorkerParams
    {
      std::string _user;    // user name under which the workers are started
      std::string _command; // command name to grep for number of workers
      int _expectedCount;   // expected number of workers running on the node
    };
    WorkerParams _injectWorkers;
    WorkerParams _copyWorkers;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used to enable sentinel alarms
   */
  struct AlarmParams
  {
    bool _isProductionSystem;      // indicates if the SM is running in production system.
                                   // If set to false, certain checks/alarms are disabled. 
    unsigned int _errorEvents;     // number of recent error events triggering an alarm
    unsigned int _unwantedEvents;  // number of unwanted events triggering an alarm

    AlarmParams() :
    _isProductionSystem(false) {}; // Initialize default to false here. This struct is
                                   // used before the actual values are set during the
                                   // configure transition.
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
   * $Revision: 1.23 $
   * $Date: 2010/09/28 16:25:29 $
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
    void actionPerformed(xdata::Event& isEvt);

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

    struct DiskWritingParams _diskWriteParamCopy;
    struct DQMProcessingParams _dqmParamCopy;
    struct EventServingParams _eventServeParamCopy;
    struct QueueConfigurationParams _queueConfigParamCopy;
    struct WorkerThreadParams _workerThreadParamCopy;
    struct ResourceMonitorParams _resourceMonitorParamCopy;
    struct AlarmParams _alarmParamCopy;

    mutable boost::mutex _generalMutex;

    std::string _previousStreamCfg;
    bool _streamConfigurationChanged;

    xdata::UnsignedInteger32 _infospaceRunNumber;
    unsigned int _localRunNumber;

    xdata::String _streamConfiguration;
    xdata::String _fileName;
    xdata::String _filePath;
    xdata::String _dbFilePath;
    xdata::Vector<xdata::String> _otherDiskPaths;
    xdata::String _fileCatalog;
    xdata::String _setupLabel;
    xdata::Integer _nLogicalDisk;
    xdata::Integer _maxFileSize;
    xdata::Double _highWaterMark;
    xdata::Double _failHighWaterMark;
    xdata::Double _lumiSectionTimeOut;
    xdata::Integer _fileClosingTestInterval;
    xdata::Double _fileSizeTolerance;
    xdata::String _faultyEventsStream;

    xdata::Integer _activeConsumerTimeout;  // seconds
    xdata::Integer _consumerQueueSize;
    xdata::String  _consumerQueuePolicy;
    xdata::Integer _DQMactiveConsumerTimeout;  // seconds
    xdata::Integer _DQMconsumerQueueSize;
    xdata::String  _DQMconsumerQueuePolicy;

    xdata::Boolean _collateDQM;
    xdata::Boolean _archiveDQM;
    xdata::Integer _archiveIntervalDQM;
    xdata::String  _filePrefixDQM;
    xdata::Integer _purgeTimeDQM;
    xdata::Integer _readyTimeDQM;
    xdata::Boolean _useCompressionDQM;
    xdata::Integer _compressionLevelDQM;

    xdata::UnsignedInteger32 _commandQueueSize;
    xdata::UnsignedInteger32 _dqmEventQueueSize;
    xdata::UnsignedInteger32 _dqmEventQueueMemoryLimitMB;
    xdata::UnsignedInteger32 _fragmentQueueSize;
    xdata::UnsignedInteger32 _fragmentQueueMemoryLimitMB;
    xdata::UnsignedInteger32 _registrationQueueSize;
    xdata::UnsignedInteger32 _streamQueueSize;
    xdata::UnsignedInteger32 _streamQueueMemoryLimitMB;

    xdata::Double _FPdeqWaitTime;
    xdata::Double _DWdeqWaitTime;
    xdata::Double _DQMEPdeqWaitTime;
    xdata::Double _staleFragmentTimeOut;
    xdata::Double _monitoringSleepSec;
    xdata::UnsignedInteger32 _throuphputAveragingCycles;

    xdata::String _sataUser;
    xdata::String _injectWorkersUser;
    xdata::String _injectWorkersCommand;
    xdata::Integer _nInjectWorkers;
    xdata::String _copyWorkersUser;
    xdata::String _copyWorkersCommand;
    xdata::Integer _nCopyWorkers;

    xdata::Boolean _isProductionSystem;
    xdata::UnsignedInteger32 _errorEvents;
    xdata::UnsignedInteger32 _unwantedEvents;

    mutable boost::mutex _evtStrCfgMutex;
    mutable boost::mutex _errStrCfgMutex;

    EvtStrConfigListPtr _currentEventStreamConfig;
    ErrStrConfigListPtr _currentErrorStreamConfig;
  };

}

#endif // EventFilter_StorageManager_Configuration_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

