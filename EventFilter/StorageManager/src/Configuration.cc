// $Id: Configuration.cc,v 1.40 2010/12/14 12:56:52 mommsen Exp $
/// @file: Configuration.cc

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <toolbox/net/Utils.h>

#include <sstream>


namespace stor
{
  Configuration::Configuration(xdata::InfoSpace* infoSpace,
                               unsigned long instanceNumber) :
    _streamConfigurationChanged(false),
    _infospaceRunNumber(0),
    _localRunNumber(0)
  {
    // default values are used to initialize infospace values,
    // so they should be set first
    setDiskWritingDefaults(instanceNumber);
    setDQMProcessingDefaults();
    setEventServingDefaults();
    setQueueConfigurationDefaults();
    setWorkerThreadDefaults();
    setResourceMonitorDefaults();
    setAlarmDefaults();

    setupDiskWritingInfoSpaceParams(infoSpace);
    setupDQMProcessingInfoSpaceParams(infoSpace);
    setupEventServingInfoSpaceParams(infoSpace);
    setupQueueConfigurationInfoSpaceParams(infoSpace);
    setupWorkerThreadInfoSpaceParams(infoSpace);
    setupResourceMonitorInfoSpaceParams(infoSpace);
    setupAlarmInfoSpaceParams(infoSpace);
  }

  struct DiskWritingParams Configuration::getDiskWritingParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _diskWriteParamCopy;
  }


  struct DQMProcessingParams Configuration::getDQMProcessingParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _dqmParamCopy;
  }

  struct EventServingParams Configuration::getEventServingParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _eventServeParamCopy;
  }

  struct QueueConfigurationParams Configuration::getQueueConfigurationParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _queueConfigParamCopy;
  }

  struct WorkerThreadParams Configuration::getWorkerThreadParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _workerThreadParamCopy;
  }

  struct ResourceMonitorParams Configuration::getResourceMonitorParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _resourceMonitorParamCopy;
  }

  struct AlarmParams Configuration::getAlarmParams() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _alarmParamCopy;
  }

  void Configuration::updateAllParams()
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    updateLocalDiskWritingData();
    updateLocalDQMProcessingData();
    updateLocalEventServingData();
    updateLocalQueueConfigurationData();
    updateLocalWorkerThreadData();
    updateLocalResourceMonitorData();
    updateLocalAlarmData();
    updateLocalRunNumberData();
  }

  unsigned int Configuration::getRunNumber() const
  {
    return _localRunNumber;
  }

  void Configuration::updateDiskWritingParams()
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    updateLocalDiskWritingData();
  }

  void Configuration::updateRunParams()
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    updateLocalRunNumberData();
  }

  bool Configuration::streamConfigurationHasChanged() const
  {
    boost::mutex::scoped_lock sl(_generalMutex);
    return _streamConfigurationChanged;
  }

  void Configuration::actionPerformed(xdata::Event& ispaceEvent)
  {
    boost::mutex::scoped_lock sl(_generalMutex);

    if (ispaceEvent.type() == "ItemChangedEvent")
      {
        std::string item =
          dynamic_cast<xdata::ItemChangedEvent&>(ispaceEvent).itemName();
        if (item == "STparameterSet")
          {
            evf::ParameterSetRetriever smpset(_streamConfiguration);
            std::string tmpStreamConfiguration = smpset.getAsString();

            if (tmpStreamConfiguration != _previousStreamCfg)
              {
                _streamConfigurationChanged = true;
                _previousStreamCfg = tmpStreamConfiguration;
              }
          }
      }
  }

  void Configuration::setCurrentEventStreamConfig(EvtStrConfigListPtr cfgList)
  {
    boost::mutex::scoped_lock sl(_evtStrCfgMutex);
    _currentEventStreamConfig = cfgList;
  }

  void Configuration::setCurrentErrorStreamConfig(ErrStrConfigListPtr cfgList)
  {
    boost::mutex::scoped_lock sl(_errStrCfgMutex);
    _currentErrorStreamConfig = cfgList;
  }

  EvtStrConfigListPtr Configuration::getCurrentEventStreamConfig() const
  {
    boost::mutex::scoped_lock sl(_evtStrCfgMutex);
    return _currentEventStreamConfig;
  }

  ErrStrConfigListPtr Configuration::getCurrentErrorStreamConfig() const
  {
    boost::mutex::scoped_lock sl(_errStrCfgMutex);
    return _currentErrorStreamConfig;
  }

  void Configuration::setDiskWritingDefaults(unsigned long instanceNumber)
  {
    _diskWriteParamCopy._streamConfiguration = "";
    _diskWriteParamCopy._fileName = "storageManager";
    _diskWriteParamCopy._filePath = "/tmp";
    _diskWriteParamCopy._dbFilePath = ""; // use default _filePath+"/log"
    _diskWriteParamCopy._otherDiskPaths.clear();
    _diskWriteParamCopy._fileCatalog = "summaryCatalog.txt";
    _diskWriteParamCopy._setupLabel = "Data";
    _diskWriteParamCopy._nLogicalDisk = 0;
    _diskWriteParamCopy._maxFileSizeMB = 0;
    _diskWriteParamCopy._highWaterMark = 90;
    _diskWriteParamCopy._failHighWaterMark = 95;
    _diskWriteParamCopy._lumiSectionTimeOut = boost::posix_time::seconds(45);
    _diskWriteParamCopy._fileClosingTestInterval = boost::posix_time::seconds(5);
    _diskWriteParamCopy._fileSizeTolerance = 0.0;
    _diskWriteParamCopy._faultyEventsStream = "";

    _previousStreamCfg = _diskWriteParamCopy._streamConfiguration;

    std::ostringstream oss;
    oss << instanceNumber;
    _diskWriteParamCopy._smInstanceString = oss.str();

    std::string tmpString(toolbox::net::getHostName());
    // strip domainame
    std::string::size_type pos = tmpString.find('.');  
    if (pos != std::string::npos) {  
      std::string basename = tmpString.substr(0,pos);  
      tmpString = basename;
    }
    _diskWriteParamCopy._hostName = tmpString;

    _diskWriteParamCopy._initialSafetyLevel = 0;
  }

  void Configuration::setDQMProcessingDefaults()
  {
    _dqmParamCopy._collateDQM = false;
    _dqmParamCopy._archiveDQM = false;
    _dqmParamCopy._filePrefixDQM = "/tmp/DQM";
    _dqmParamCopy._archiveIntervalDQM = 0;
    _dqmParamCopy._purgeTimeDQM = boost::posix_time::seconds(300);
    _dqmParamCopy._readyTimeDQM = boost::posix_time::seconds(120);
    _dqmParamCopy._useCompressionDQM = true;
    _dqmParamCopy._compressionLevelDQM = 1;
  }

  void Configuration::setEventServingDefaults()
  {
    _eventServeParamCopy._activeConsumerTimeout = boost::posix_time::seconds(60);
    _eventServeParamCopy._consumerQueueSize = 5;
    _eventServeParamCopy._consumerQueuePolicy = "DiscardOld";
    _eventServeParamCopy._DQMactiveConsumerTimeout = boost::posix_time::seconds(60);
    _eventServeParamCopy._DQMconsumerQueueSize = 15;
    _eventServeParamCopy._DQMconsumerQueuePolicy = "DiscardOld";
  }

  void Configuration::setQueueConfigurationDefaults()
  {
    _queueConfigParamCopy._commandQueueSize = 128;
    _queueConfigParamCopy._dqmEventQueueSize = 3072;
    _queueConfigParamCopy._dqmEventQueueMemoryLimitMB = 9999;
    _queueConfigParamCopy._fragmentQueueSize = 1024;
    _queueConfigParamCopy._fragmentQueueMemoryLimitMB = 9999;
    _queueConfigParamCopy._registrationQueueSize = 128;
    _queueConfigParamCopy._streamQueueSize = 2048;
    _queueConfigParamCopy._streamQueueMemoryLimitMB = 9999;
  }

  void Configuration::setWorkerThreadDefaults()
  {
    // set defaults
    _workerThreadParamCopy._FPdeqWaitTime = boost::posix_time::millisec(250);
    _workerThreadParamCopy._DWdeqWaitTime = boost::posix_time::millisec(500);
    _workerThreadParamCopy._DQMEPdeqWaitTime = boost::posix_time::millisec(500);
  
    _workerThreadParamCopy._staleFragmentTimeOut = boost::posix_time::seconds(60);
    _workerThreadParamCopy._monitoringSleepSec = boost::posix_time::seconds(1);
    _workerThreadParamCopy._throuphputAveragingCycles = 10;
  }

  void Configuration::setResourceMonitorDefaults()
  {
    // set defaults
    _resourceMonitorParamCopy._sataUser = "";
    _resourceMonitorParamCopy._injectWorkers._user = "smpro";
    _resourceMonitorParamCopy._injectWorkers._command = "/InjectWorker.pl /store/global";
    _resourceMonitorParamCopy._injectWorkers._expectedCount = -1;
    _resourceMonitorParamCopy._copyWorkers._user = "cmsprod";
    _resourceMonitorParamCopy._copyWorkers._command = "CopyManager/CopyWorker.pl";
    _resourceMonitorParamCopy._copyWorkers._expectedCount = -1;
  }

  void Configuration::setAlarmDefaults()
  {
    // set defaults
    _alarmParamCopy._isProductionSystem = false;
    _alarmParamCopy._errorEvents = 10;
    _alarmParamCopy._unwantedEvents = 10000;
  }

  void Configuration::
  setupDiskWritingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults into the xdata variables
    _streamConfiguration = _diskWriteParamCopy._streamConfiguration;
    _fileName = _diskWriteParamCopy._fileName;
    _filePath = _diskWriteParamCopy._filePath;
    _dbFilePath = _diskWriteParamCopy._dbFilePath;
    _fileCatalog = _diskWriteParamCopy._fileCatalog;
    _setupLabel = _diskWriteParamCopy._setupLabel;
    _nLogicalDisk = _diskWriteParamCopy._nLogicalDisk;
    _maxFileSize = _diskWriteParamCopy._maxFileSizeMB;
    _highWaterMark = _diskWriteParamCopy._highWaterMark;
    _failHighWaterMark = _diskWriteParamCopy._failHighWaterMark;
    _lumiSectionTimeOut = utils::duration_to_seconds(_diskWriteParamCopy._lumiSectionTimeOut);
    _fileClosingTestInterval = _diskWriteParamCopy._fileClosingTestInterval.total_seconds();
    _fileSizeTolerance = _diskWriteParamCopy._fileSizeTolerance;
    _faultyEventsStream = _diskWriteParamCopy._faultyEventsStream;

    utils::getXdataVector(_diskWriteParamCopy._otherDiskPaths, _otherDiskPaths);


    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("STparameterSet", &_streamConfiguration);
    infoSpace->fireItemAvailable("fileName", &_fileName);
    infoSpace->fireItemAvailable("filePath", &_filePath);
    infoSpace->fireItemAvailable("dbFilePath", &_dbFilePath);
    infoSpace->fireItemAvailable("otherDiskPaths", &_otherDiskPaths);
    infoSpace->fireItemAvailable("fileCatalog", &_fileCatalog);
    infoSpace->fireItemAvailable("setupLabel", &_setupLabel);
    infoSpace->fireItemAvailable("nLogicalDisk", &_nLogicalDisk);
    infoSpace->fireItemAvailable("maxFileSize", &_maxFileSize);
    infoSpace->fireItemAvailable("highWaterMark", &_highWaterMark);
    infoSpace->fireItemAvailable("failHighWaterMark", &_failHighWaterMark);
    infoSpace->fireItemAvailable("lumiSectionTimeOut", &_lumiSectionTimeOut);
    infoSpace->fireItemAvailable("fileClosingTestInterval",
                                 &_fileClosingTestInterval);
    infoSpace->fireItemAvailable("fileSizeTolerance", &_fileSizeTolerance);
    infoSpace->fireItemAvailable("faultyEventsStream", &_faultyEventsStream);

    // special handling for the stream configuration string (we
    // want to note when it changes to see if we need to reconfigure
    // between runs)
    infoSpace->addItemChangedListener("STparameterSet", this);
  }

  void Configuration::
  setupDQMProcessingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _collateDQM = _dqmParamCopy._collateDQM;
    _archiveDQM = _dqmParamCopy._archiveDQM;
    _archiveIntervalDQM = _dqmParamCopy._archiveIntervalDQM;
    _filePrefixDQM = _dqmParamCopy._filePrefixDQM;
    _purgeTimeDQM = _dqmParamCopy._purgeTimeDQM.total_seconds();
    _readyTimeDQM = _dqmParamCopy._readyTimeDQM.total_seconds();
    _useCompressionDQM = _dqmParamCopy._useCompressionDQM;
    _compressionLevelDQM = _dqmParamCopy._compressionLevelDQM;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("collateDQM", &_collateDQM);
    infoSpace->fireItemAvailable("archiveDQM", &_archiveDQM);
    infoSpace->fireItemAvailable("archiveIntervalDQM", &_archiveIntervalDQM);
    infoSpace->fireItemAvailable("purgeTimeDQM", &_purgeTimeDQM);
    infoSpace->fireItemAvailable("readyTimeDQM", &_readyTimeDQM);
    infoSpace->fireItemAvailable("filePrefixDQM", &_filePrefixDQM);
    infoSpace->fireItemAvailable("useCompressionDQM", &_useCompressionDQM);
    infoSpace->fireItemAvailable("compressionLevelDQM", &_compressionLevelDQM);
  }

  void Configuration::
  setupEventServingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _activeConsumerTimeout = _eventServeParamCopy._activeConsumerTimeout.total_seconds();
    _consumerQueueSize = _eventServeParamCopy._consumerQueueSize;
    _consumerQueuePolicy = _eventServeParamCopy._consumerQueuePolicy;
    _DQMactiveConsumerTimeout = _eventServeParamCopy._DQMactiveConsumerTimeout.total_seconds();
    _DQMconsumerQueueSize = _eventServeParamCopy._DQMconsumerQueueSize;
    _DQMconsumerQueuePolicy = _eventServeParamCopy._DQMconsumerQueuePolicy;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable( "runNumber", &_infospaceRunNumber );
    infoSpace->fireItemAvailable("activeConsumerTimeout",
                                 &_activeConsumerTimeout);
    infoSpace->fireItemAvailable("consumerQueueSize",&_consumerQueueSize);
    infoSpace->fireItemAvailable("consumerQueuePolicy",&_consumerQueuePolicy);
    infoSpace->fireItemAvailable("DQMactiveConsumerTimeout",
                              &_DQMactiveConsumerTimeout);
    infoSpace->fireItemAvailable("DQMconsumerQueueSize",
                                 &_DQMconsumerQueueSize);
    infoSpace->fireItemAvailable("DQMconsumerQueuePolicy",&_DQMconsumerQueuePolicy);
  }

  void Configuration::
  setupQueueConfigurationInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _commandQueueSize = _queueConfigParamCopy._commandQueueSize;
    _dqmEventQueueSize = _queueConfigParamCopy._dqmEventQueueSize;
    _dqmEventQueueMemoryLimitMB = _queueConfigParamCopy._dqmEventQueueMemoryLimitMB;
    _fragmentQueueSize = _queueConfigParamCopy._fragmentQueueSize;
    _fragmentQueueMemoryLimitMB = _queueConfigParamCopy._fragmentQueueMemoryLimitMB;
    _registrationQueueSize = _queueConfigParamCopy._registrationQueueSize;
    _streamQueueSize = _queueConfigParamCopy._streamQueueSize;
    _streamQueueMemoryLimitMB = _queueConfigParamCopy._streamQueueMemoryLimitMB;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("commandQueueSize", &_commandQueueSize);
    infoSpace->fireItemAvailable("dqmEventQueueSize", &_dqmEventQueueSize);
    infoSpace->fireItemAvailable("dqmEventQueueMemoryLimitMB", &_dqmEventQueueMemoryLimitMB);
    infoSpace->fireItemAvailable("fragmentQueueSize", &_fragmentQueueSize);
    infoSpace->fireItemAvailable("fragmentQueueMemoryLimitMB", &_fragmentQueueMemoryLimitMB);
    infoSpace->fireItemAvailable("registrationQueueSize",
                                 &_registrationQueueSize);
    infoSpace->fireItemAvailable("streamQueueSize", &_streamQueueSize);
    infoSpace->fireItemAvailable("streamQueueMemoryLimitMB", &_streamQueueMemoryLimitMB);
  }

  void Configuration::
  setupWorkerThreadInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _FPdeqWaitTime = utils::duration_to_seconds(_workerThreadParamCopy._FPdeqWaitTime);
    _DWdeqWaitTime = utils::duration_to_seconds(_workerThreadParamCopy._DWdeqWaitTime);
    _DQMEPdeqWaitTime = utils::duration_to_seconds(_workerThreadParamCopy._DQMEPdeqWaitTime);
    _staleFragmentTimeOut = utils::duration_to_seconds(_workerThreadParamCopy._staleFragmentTimeOut);
    _monitoringSleepSec = utils::duration_to_seconds(_workerThreadParamCopy._monitoringSleepSec);
    _throuphputAveragingCycles = _workerThreadParamCopy._throuphputAveragingCycles;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("FPdeqWaitTime", &_FPdeqWaitTime);
    infoSpace->fireItemAvailable("DWdeqWaitTime", &_DWdeqWaitTime);
    infoSpace->fireItemAvailable("DQMEPdeqWaitTime", &_DQMEPdeqWaitTime);
    infoSpace->fireItemAvailable("staleFragmentTimeOut", &_staleFragmentTimeOut);
    infoSpace->fireItemAvailable("monitoringSleepSec", &_monitoringSleepSec);
    infoSpace->fireItemAvailable("throuphputAveragingCycles", &_throuphputAveragingCycles);
  }

  void Configuration::
  setupResourceMonitorInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _sataUser = _resourceMonitorParamCopy._sataUser;
    _injectWorkersUser = _resourceMonitorParamCopy._injectWorkers._user;
    _injectWorkersCommand = _resourceMonitorParamCopy._injectWorkers._command;
    _nInjectWorkers = _resourceMonitorParamCopy._injectWorkers._expectedCount;
    _copyWorkersUser = _resourceMonitorParamCopy._copyWorkers._user;
    _copyWorkersCommand = _resourceMonitorParamCopy._copyWorkers._command;
    _nCopyWorkers = _resourceMonitorParamCopy._copyWorkers._expectedCount;
 
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("sataUser", &_sataUser);
    infoSpace->fireItemAvailable("injectWorkersUser", &_injectWorkersUser);
    infoSpace->fireItemAvailable("injectWorkersCommand", &_injectWorkersCommand);
    infoSpace->fireItemAvailable("nInjectWorkers", &_nInjectWorkers);
    infoSpace->fireItemAvailable("copyWorkersUser", &_copyWorkersUser);
    infoSpace->fireItemAvailable("copyWorkersCommand", &_copyWorkersCommand);
    infoSpace->fireItemAvailable("nCopyWorkers", &_nCopyWorkers);
  }

  void Configuration::
  setupAlarmInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    _isProductionSystem = _alarmParamCopy._isProductionSystem;
    _errorEvents = _alarmParamCopy._errorEvents;
    _unwantedEvents = _alarmParamCopy._unwantedEvents;
 
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("isProductionSystem", &_isProductionSystem);
    infoSpace->fireItemAvailable("errorEvents", &_errorEvents);
    infoSpace->fireItemAvailable("unwantedEvents", &_unwantedEvents);
  }

  void Configuration::updateLocalDiskWritingData()
  {
    evf::ParameterSetRetriever smpset(_streamConfiguration);
    _diskWriteParamCopy._streamConfiguration = smpset.getAsString();

    _diskWriteParamCopy._fileName = _fileName;
    _diskWriteParamCopy._filePath = _filePath;
    if ( _dbFilePath.value_.empty() )
      _diskWriteParamCopy._dbFilePath = _filePath.value_ + "/log";
    else
      _diskWriteParamCopy._dbFilePath = _dbFilePath;
    _diskWriteParamCopy._fileCatalog = _fileCatalog;
    _diskWriteParamCopy._setupLabel = _setupLabel;
    _diskWriteParamCopy._nLogicalDisk = _nLogicalDisk;
    _diskWriteParamCopy._maxFileSizeMB = _maxFileSize;
    _diskWriteParamCopy._highWaterMark = _highWaterMark;
    _diskWriteParamCopy._failHighWaterMark = _failHighWaterMark;
    _diskWriteParamCopy._lumiSectionTimeOut = utils::seconds_to_duration(_lumiSectionTimeOut);
    _diskWriteParamCopy._fileClosingTestInterval =
      boost::posix_time::seconds( static_cast<int>(_fileClosingTestInterval) );
    _diskWriteParamCopy._fileSizeTolerance = _fileSizeTolerance;
    _diskWriteParamCopy._faultyEventsStream = _faultyEventsStream;

    utils::getStdVector(_otherDiskPaths, _diskWriteParamCopy._otherDiskPaths);


    _streamConfigurationChanged = false;
  }

  void Configuration::updateLocalDQMProcessingData()
  {
    _dqmParamCopy._collateDQM = _collateDQM;
    _dqmParamCopy._archiveDQM = _archiveDQM;
    _dqmParamCopy._archiveIntervalDQM = _archiveIntervalDQM;
    _dqmParamCopy._filePrefixDQM = _filePrefixDQM;
    _dqmParamCopy._purgeTimeDQM =
      boost::posix_time::seconds( static_cast<int>(_purgeTimeDQM) );
    _dqmParamCopy._readyTimeDQM =
      boost::posix_time::seconds( static_cast<int>(_readyTimeDQM) );
    _dqmParamCopy._useCompressionDQM = _useCompressionDQM;
    _dqmParamCopy._compressionLevelDQM = _compressionLevelDQM;

    // make sure that purge time is larger than ready time
    if ( _dqmParamCopy._purgeTimeDQM < _dqmParamCopy._readyTimeDQM )
    {
      _dqmParamCopy._purgeTimeDQM = _dqmParamCopy._readyTimeDQM + boost::posix_time::seconds(10);
    }
  }

  void Configuration::updateLocalEventServingData()
  {
    _eventServeParamCopy._activeConsumerTimeout =
      boost::posix_time::seconds( static_cast<int>(_activeConsumerTimeout) );
    _eventServeParamCopy._consumerQueueSize = _consumerQueueSize;
    _eventServeParamCopy._consumerQueuePolicy = _consumerQueuePolicy;
    _eventServeParamCopy._DQMactiveConsumerTimeout = 
      boost::posix_time::seconds( static_cast<int>(_DQMactiveConsumerTimeout) );
    _eventServeParamCopy._DQMconsumerQueueSize = _DQMconsumerQueueSize;
    _eventServeParamCopy._DQMconsumerQueuePolicy = _DQMconsumerQueuePolicy;

    // validation
    if (_eventServeParamCopy._consumerQueueSize < 1)
      {
        _eventServeParamCopy._consumerQueueSize = 1;
      }
    if (_eventServeParamCopy._DQMconsumerQueueSize < 1)
      {
        _eventServeParamCopy._DQMconsumerQueueSize = 1;
      }
  }

  void Configuration::updateLocalQueueConfigurationData()
  {
    _queueConfigParamCopy._commandQueueSize = _commandQueueSize;
    _queueConfigParamCopy._dqmEventQueueSize = _dqmEventQueueSize;
    _queueConfigParamCopy._dqmEventQueueMemoryLimitMB = _dqmEventQueueMemoryLimitMB;
    _queueConfigParamCopy._fragmentQueueSize = _fragmentQueueSize;
    _queueConfigParamCopy._fragmentQueueMemoryLimitMB = _fragmentQueueMemoryLimitMB;
    _queueConfigParamCopy._registrationQueueSize = _registrationQueueSize;
    _queueConfigParamCopy._streamQueueSize = _streamQueueSize;
    _queueConfigParamCopy._streamQueueMemoryLimitMB = _streamQueueMemoryLimitMB;
  }

  void Configuration::updateLocalWorkerThreadData()
  {
    _workerThreadParamCopy._FPdeqWaitTime = utils::seconds_to_duration(_FPdeqWaitTime);
    _workerThreadParamCopy._DWdeqWaitTime = utils::seconds_to_duration(_DWdeqWaitTime);
    _workerThreadParamCopy._DQMEPdeqWaitTime = utils::seconds_to_duration(_DQMEPdeqWaitTime);

    _workerThreadParamCopy._staleFragmentTimeOut = utils::seconds_to_duration(_staleFragmentTimeOut);
    _workerThreadParamCopy._monitoringSleepSec = utils::seconds_to_duration(_monitoringSleepSec);
    _workerThreadParamCopy._throuphputAveragingCycles = _throuphputAveragingCycles;
  }

  void Configuration::updateLocalResourceMonitorData()
  {
    _resourceMonitorParamCopy._sataUser = _sataUser;
    _resourceMonitorParamCopy._injectWorkers._user = _injectWorkersUser;
    _resourceMonitorParamCopy._injectWorkers._command = _injectWorkersCommand;
    _resourceMonitorParamCopy._injectWorkers._expectedCount = _nInjectWorkers;
    _resourceMonitorParamCopy._copyWorkers._user = _copyWorkersUser;
    _resourceMonitorParamCopy._copyWorkers._command = _copyWorkersCommand;
    _resourceMonitorParamCopy._copyWorkers._expectedCount = _nCopyWorkers;
  }

  void Configuration::updateLocalAlarmData()
  {
    _alarmParamCopy._isProductionSystem = _isProductionSystem;
    _alarmParamCopy._errorEvents = _errorEvents;
    _alarmParamCopy._unwantedEvents = _unwantedEvents;
  }

  void Configuration::updateLocalRunNumberData()
  {
    _localRunNumber = _infospaceRunNumber;
  }

  void parseStreamConfiguration(std::string cfgString,
                                EvtStrConfigListPtr evtCfgList,
                                ErrStrConfigListPtr errCfgList)
  {
    if (cfgString == "") return;

    PythonProcessDesc py_pdesc(cfgString.c_str());
    boost::shared_ptr<edm::ProcessDesc> pdesc = py_pdesc.processDesc();
    boost::shared_ptr<edm::ParameterSet> smPSet = pdesc->getProcessPSet();

    // loop over each end path
    std::vector<std::string> allEndPaths = 
      smPSet->getParameter<std::vector<std::string> >("@end_paths");
    for(std::vector<std::string>::iterator endPathIter = allEndPaths.begin();
        endPathIter != allEndPaths.end(); ++endPathIter) {

      // loop over each element in the end path list (not sure why...)
      std::vector<std::string> anEndPath =
        smPSet->getParameter<std::vector<std::string> >((*endPathIter));
      for(std::vector<std::string>::iterator ep2Iter = anEndPath.begin();
          ep2Iter != anEndPath.end(); ++ep2Iter) {

        // fetch the end path parameter set
        edm::ParameterSet endPathPSet =
          smPSet->getParameter<edm::ParameterSet>((*ep2Iter));
        if (! endPathPSet.empty()) {
          std::string mod_type =
            endPathPSet.getParameter<std::string> ("@module_type");
          if (mod_type == "EventStreamFileWriter") {

            std::string streamLabel =
              endPathPSet.getParameter<std::string> ("streamLabel");
            int maxFileSizeMB = endPathPSet.getParameter<int> ("maxSize");
            std::string newRequestedEvents = endPathPSet.getUntrackedParameter("TriggerSelector",std::string());
            Strings requestedEvents =
              edm::EventSelector::getEventSelectionVString(endPathPSet);
            std::string requestedOMLabel =
              endPathPSet.getUntrackedParameter<std::string>("SelectHLTOutput",
                                                             std::string());
            double fractionToDisk =
              endPathPSet.getUntrackedParameter<double>("fractionToDisk", 1);

            EventStreamConfigurationInfo cfgInfo(streamLabel,
                                                 maxFileSizeMB,
                                                 newRequestedEvents,
                                                 requestedEvents,
                                                 requestedOMLabel,
                                                 fractionToDisk);
            evtCfgList->push_back(cfgInfo);
          }
          else if (mod_type == "ErrorStreamFileWriter" ||
                   mod_type == "FRDStreamFileWriter") {

            std::string streamLabel =
              endPathPSet.getParameter<std::string> ("streamLabel");
            int maxFileSizeMB = endPathPSet.getParameter<int> ("maxSize");

            ErrorStreamConfigurationInfo cfgInfo(streamLabel,
                                                 maxFileSizeMB);
            errCfgList->push_back(cfgInfo);
          }
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
