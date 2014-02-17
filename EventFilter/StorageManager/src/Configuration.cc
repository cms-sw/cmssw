// $Id: Configuration.cc,v 1.47 2012/06/08 10:20:33 mommsen Exp $
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
    streamConfigurationChanged_(false),
    infospaceRunNumber_(0),
    localRunNumber_(0)
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
    boost::mutex::scoped_lock sl(generalMutex_);
    return diskWriteParamCopy_;
  }

  struct DQMProcessingParams Configuration::getDQMProcessingParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return dqmParamCopy_;
  }

  struct EventServingParams Configuration::getEventServingParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return eventServeParamCopy_;
  }

  struct QueueConfigurationParams Configuration::getQueueConfigurationParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return queueConfigParamCopy_;
  }

  struct WorkerThreadParams Configuration::getWorkerThreadParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return workerThreadParamCopy_;
  }

  struct ResourceMonitorParams Configuration::getResourceMonitorParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return resourceMonitorParamCopy_;
  }

  struct AlarmParams Configuration::getAlarmParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return alarmParamCopy_;
  }

  void Configuration::updateAllParams()
  {
    boost::mutex::scoped_lock sl(generalMutex_);
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
    return localRunNumber_;
  }

  void Configuration::updateDiskWritingParams()
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    updateLocalDiskWritingData();
  }

  void Configuration::updateRunParams()
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    updateLocalRunNumberData();
  }

  bool Configuration::streamConfigurationHasChanged() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return streamConfigurationChanged_;
  }

  void Configuration::actionPerformed(xdata::Event& ispaceEvent)
  {
    boost::mutex::scoped_lock sl(generalMutex_);

    if (ispaceEvent.type() == "ItemChangedEvent")
      {
        std::string item =
          dynamic_cast<xdata::ItemChangedEvent&>(ispaceEvent).itemName();
        if (item == "STparameterSet")
          {
            evf::ParameterSetRetriever smpset(streamConfiguration_);
            std::string tmpStreamConfiguration = smpset.getAsString();

            if (tmpStreamConfiguration != previousStreamCfg_)
              {
                streamConfigurationChanged_ = true;
                previousStreamCfg_ = tmpStreamConfiguration;
              }
          }
      }
  }

  void Configuration::setCurrentEventStreamConfig(EvtStrConfigListPtr cfgList)
  {
    boost::mutex::scoped_lock sl(evtStrCfgMutex_);
    currentEventStreamConfig_ = cfgList;
  }

  void Configuration::setCurrentErrorStreamConfig(ErrStrConfigListPtr cfgList)
  {
    boost::mutex::scoped_lock sl(errStrCfgMutex_);
    currentErrorStreamConfig_ = cfgList;
  }

  EvtStrConfigListPtr Configuration::getCurrentEventStreamConfig() const
  {
    boost::mutex::scoped_lock sl(evtStrCfgMutex_);
    return currentEventStreamConfig_;
  }

  ErrStrConfigListPtr Configuration::getCurrentErrorStreamConfig() const
  {
    boost::mutex::scoped_lock sl(errStrCfgMutex_);
    return currentErrorStreamConfig_;
  }

  void Configuration::setDiskWritingDefaults(unsigned long instanceNumber)
  {
    diskWriteParamCopy_.streamConfiguration_ = "";
    diskWriteParamCopy_.fileName_ = "storageManager";
    diskWriteParamCopy_.filePath_ = "/tmp";
    diskWriteParamCopy_.dbFilePath_ = ""; // use default filePath_+"/log"
    diskWriteParamCopy_.otherDiskPaths_.clear();
    diskWriteParamCopy_.setupLabel_ = "Data";
    diskWriteParamCopy_.nLogicalDisk_ = 0;
    diskWriteParamCopy_.maxFileSizeMB_ = 0;
    diskWriteParamCopy_.highWaterMark_ = 90;
    diskWriteParamCopy_.failHighWaterMark_ = 95;
    diskWriteParamCopy_.lumiSectionTimeOut_ = boost::posix_time::seconds(45);
    diskWriteParamCopy_.fileClosingTestInterval_ = boost::posix_time::seconds(5);
    diskWriteParamCopy_.fileSizeTolerance_ = 0.0;
    diskWriteParamCopy_.faultyEventsStream_ = "";
    diskWriteParamCopy_.checkAdler32_ = false;

    previousStreamCfg_ = diskWriteParamCopy_.streamConfiguration_;

    std::ostringstream oss;
    oss << instanceNumber;
    diskWriteParamCopy_.smInstanceString_ = oss.str();

    std::string tmpString(toolbox::net::getHostName());
    // strip domainame
    std::string::size_type pos = tmpString.find('.');  
    if (pos != std::string::npos) {  
      std::string basename = tmpString.substr(0,pos);  
      tmpString = basename;
    }
    diskWriteParamCopy_.hostName_ = tmpString;

    diskWriteParamCopy_.initialSafetyLevel_ = 0;
  }

  void Configuration::setDQMProcessingDefaults()
  {
    dqmParamCopy_.collateDQM_ = true;
    dqmParamCopy_.readyTimeDQM_ = boost::posix_time::seconds(120);
    dqmParamCopy_.useCompressionDQM_ = true;
    dqmParamCopy_.compressionLevelDQM_ = 1;
    dqmParamCopy_.discardDQMUpdatesForOlderLS_ = 16;
  }

  void Configuration::setEventServingDefaults()
  {
    eventServeParamCopy_.activeConsumerTimeout_ = boost::posix_time::seconds(60);
    eventServeParamCopy_.consumerQueueSize_ = 10;
    eventServeParamCopy_.consumerQueuePolicy_ = "DiscardOld";
    eventServeParamCopy_._DQMactiveConsumerTimeout = boost::posix_time::seconds(60);
    eventServeParamCopy_._DQMconsumerQueueSize = 15;
    eventServeParamCopy_._DQMconsumerQueuePolicy = "DiscardOld";
  }

  void Configuration::setQueueConfigurationDefaults()
  {
    queueConfigParamCopy_.commandQueueSize_ = 128;
    queueConfigParamCopy_.dqmEventQueueSize_ = 3072;
    queueConfigParamCopy_.dqmEventQueueMemoryLimitMB_ = 1024;
    queueConfigParamCopy_.fragmentQueueSize_ = 1024;
    queueConfigParamCopy_.fragmentQueueMemoryLimitMB_ = 1024;
    queueConfigParamCopy_.registrationQueueSize_ = 128;
    queueConfigParamCopy_.streamQueueSize_ = 2048;
    queueConfigParamCopy_.streamQueueMemoryLimitMB_ = 2048;
    queueConfigParamCopy_.fragmentStoreMemoryLimitMB_ = 1024;
  }

  void Configuration::setWorkerThreadDefaults()
  {
    // set defaults
    workerThreadParamCopy_.FPdeqWaitTime_ = boost::posix_time::millisec(250);
    workerThreadParamCopy_.DWdeqWaitTime_ = boost::posix_time::millisec(500);
    workerThreadParamCopy_.DQMEPdeqWaitTime_ = boost::posix_time::millisec(500);
  
    workerThreadParamCopy_.staleFragmentTimeOut_ = boost::posix_time::seconds(60);
    workerThreadParamCopy_.monitoringSleepSec_ = boost::posix_time::seconds(1);
    workerThreadParamCopy_.throuphputAveragingCycles_ = 10;
  }

  void Configuration::setResourceMonitorDefaults()
  {
    // set defaults
    resourceMonitorParamCopy_.sataUser_ = "";
    resourceMonitorParamCopy_.injectWorkers_.user_ = "smpro";
    resourceMonitorParamCopy_.injectWorkers_.command_ = "/InjectWorker.pl /store/global";
    resourceMonitorParamCopy_.injectWorkers_.expectedCount_ = -1;
    resourceMonitorParamCopy_.copyWorkers_.user_ = "cmsprod";
    resourceMonitorParamCopy_.copyWorkers_.command_ = "CopyManager/CopyWorker.pl";
    resourceMonitorParamCopy_.copyWorkers_.expectedCount_ = -1;
  }

  void Configuration::setAlarmDefaults()
  {
    // set defaults
    alarmParamCopy_.isProductionSystem_ = false;
    alarmParamCopy_.careAboutUnwantedEvents_ = true;
    alarmParamCopy_.errorEvents_ = 10;
    alarmParamCopy_.unwantedEvents_ = 10000;
  }

  void Configuration::
  setupDiskWritingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults into the xdata variables
    streamConfiguration_ = diskWriteParamCopy_.streamConfiguration_;
    fileName_ = diskWriteParamCopy_.fileName_;
    filePath_ = diskWriteParamCopy_.filePath_;
    dbFilePath_ = diskWriteParamCopy_.dbFilePath_;
    setupLabel_ = diskWriteParamCopy_.setupLabel_;
    nLogicalDisk_ = diskWriteParamCopy_.nLogicalDisk_;
    maxFileSize_ = diskWriteParamCopy_.maxFileSizeMB_;
    highWaterMark_ = diskWriteParamCopy_.highWaterMark_;
    failHighWaterMark_ = diskWriteParamCopy_.failHighWaterMark_;
    lumiSectionTimeOut_ = utils::durationToSeconds(diskWriteParamCopy_.lumiSectionTimeOut_);
    fileClosingTestInterval_ = diskWriteParamCopy_.fileClosingTestInterval_.total_seconds();
    fileSizeTolerance_ = diskWriteParamCopy_.fileSizeTolerance_;
    faultyEventsStream_ = diskWriteParamCopy_.faultyEventsStream_;
    checkAdler32_ = diskWriteParamCopy_.checkAdler32_;

    utils::getXdataVector(diskWriteParamCopy_.otherDiskPaths_, otherDiskPaths_);


    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("STparameterSet", &streamConfiguration_);
    infoSpace->fireItemAvailable("fileName", &fileName_);
    infoSpace->fireItemAvailable("filePath", &filePath_);
    infoSpace->fireItemAvailable("dbFilePath", &dbFilePath_);
    infoSpace->fireItemAvailable("otherDiskPaths", &otherDiskPaths_);
    infoSpace->fireItemAvailable("setupLabel", &setupLabel_);
    infoSpace->fireItemAvailable("nLogicalDisk", &nLogicalDisk_);
    infoSpace->fireItemAvailable("maxFileSize", &maxFileSize_);
    infoSpace->fireItemAvailable("highWaterMark", &highWaterMark_);
    infoSpace->fireItemAvailable("failHighWaterMark", &failHighWaterMark_);
    infoSpace->fireItemAvailable("lumiSectionTimeOut", &lumiSectionTimeOut_);
    infoSpace->fireItemAvailable("fileClosingTestInterval",
                                 &fileClosingTestInterval_);
    infoSpace->fireItemAvailable("fileSizeTolerance", &fileSizeTolerance_);
    infoSpace->fireItemAvailable("faultyEventsStream", &faultyEventsStream_);
    infoSpace->fireItemAvailable("checkAdler32", &checkAdler32_);

    // special handling for the stream configuration string (we
    // want to note when it changes to see if we need to reconfigure
    // between runs)
    infoSpace->addItemChangedListener("STparameterSet", this);
  }

  void Configuration::
  setupDQMProcessingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    collateDQM_ = dqmParamCopy_.collateDQM_;
    readyTimeDQM_ = dqmParamCopy_.readyTimeDQM_.total_seconds();
    useCompressionDQM_ = dqmParamCopy_.useCompressionDQM_;
    compressionLevelDQM_ = dqmParamCopy_.compressionLevelDQM_;
    discardDQMUpdatesForOlderLS_ = dqmParamCopy_.discardDQMUpdatesForOlderLS_;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("collateDQM", &collateDQM_);
    infoSpace->fireItemAvailable("readyTimeDQM", &readyTimeDQM_);
    infoSpace->fireItemAvailable("useCompressionDQM", &useCompressionDQM_);
    infoSpace->fireItemAvailable("compressionLevelDQM", &compressionLevelDQM_);
    infoSpace->fireItemAvailable("discardDQMUpdatesForOlderLS", &discardDQMUpdatesForOlderLS_);
  }

  void Configuration::
  setupEventServingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    activeConsumerTimeout_ = eventServeParamCopy_.activeConsumerTimeout_.total_seconds();
    consumerQueueSize_ = eventServeParamCopy_.consumerQueueSize_;
    consumerQueuePolicy_ = eventServeParamCopy_.consumerQueuePolicy_;
    _DQMactiveConsumerTimeout = eventServeParamCopy_._DQMactiveConsumerTimeout.total_seconds();
    _DQMconsumerQueueSize = eventServeParamCopy_._DQMconsumerQueueSize;
    _DQMconsumerQueuePolicy = eventServeParamCopy_._DQMconsumerQueuePolicy;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("runNumber", &infospaceRunNumber_);
    infoSpace->fireItemAvailable("activeConsumerTimeout",
                                 &activeConsumerTimeout_);
    infoSpace->fireItemAvailable("consumerQueueSize",&consumerQueueSize_);
    infoSpace->fireItemAvailable("consumerQueuePolicy",&consumerQueuePolicy_);
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
    commandQueueSize_ = queueConfigParamCopy_.commandQueueSize_;
    dqmEventQueueSize_ = queueConfigParamCopy_.dqmEventQueueSize_;
    dqmEventQueueMemoryLimitMB_ = queueConfigParamCopy_.dqmEventQueueMemoryLimitMB_;
    fragmentQueueSize_ = queueConfigParamCopy_.fragmentQueueSize_;
    fragmentQueueMemoryLimitMB_ = queueConfigParamCopy_.fragmentQueueMemoryLimitMB_;
    registrationQueueSize_ = queueConfigParamCopy_.registrationQueueSize_;
    streamQueueSize_ = queueConfigParamCopy_.streamQueueSize_;
    streamQueueMemoryLimitMB_ = queueConfigParamCopy_.streamQueueMemoryLimitMB_;
    fragmentStoreMemoryLimitMB_ = queueConfigParamCopy_.fragmentStoreMemoryLimitMB_;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("commandQueueSize", &commandQueueSize_);
    infoSpace->fireItemAvailable("dqmEventQueueSize", &dqmEventQueueSize_);
    infoSpace->fireItemAvailable("dqmEventQueueMemoryLimitMB", &dqmEventQueueMemoryLimitMB_);
    infoSpace->fireItemAvailable("fragmentQueueSize", &fragmentQueueSize_);
    infoSpace->fireItemAvailable("fragmentQueueMemoryLimitMB", &fragmentQueueMemoryLimitMB_);
    infoSpace->fireItemAvailable("registrationQueueSize", &registrationQueueSize_);
    infoSpace->fireItemAvailable("streamQueueSize", &streamQueueSize_);
    infoSpace->fireItemAvailable("streamQueueMemoryLimitMB", &streamQueueMemoryLimitMB_);
    infoSpace->fireItemAvailable("fragmentStoreMemoryLimitMB", &fragmentStoreMemoryLimitMB_);
  }

  void Configuration::
  setupWorkerThreadInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    FPdeqWaitTime_ = utils::durationToSeconds(workerThreadParamCopy_.FPdeqWaitTime_);
    DWdeqWaitTime_ = utils::durationToSeconds(workerThreadParamCopy_.DWdeqWaitTime_);
    DQMEPdeqWaitTime_ = utils::durationToSeconds(workerThreadParamCopy_.DQMEPdeqWaitTime_);
    staleFragmentTimeOut_ = utils::durationToSeconds(workerThreadParamCopy_.staleFragmentTimeOut_);
    monitoringSleepSec_ = utils::durationToSeconds(workerThreadParamCopy_.monitoringSleepSec_);
    throuphputAveragingCycles_ = workerThreadParamCopy_.throuphputAveragingCycles_;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("FPdeqWaitTime", &FPdeqWaitTime_);
    infoSpace->fireItemAvailable("DWdeqWaitTime", &DWdeqWaitTime_);
    infoSpace->fireItemAvailable("DQMEPdeqWaitTime", &DQMEPdeqWaitTime_);
    infoSpace->fireItemAvailable("staleFragmentTimeOut", &staleFragmentTimeOut_);
    infoSpace->fireItemAvailable("monitoringSleepSec", &monitoringSleepSec_);
    infoSpace->fireItemAvailable("throuphputAveragingCycles", &throuphputAveragingCycles_);
  }

  void Configuration::
  setupResourceMonitorInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    sataUser_ = resourceMonitorParamCopy_.sataUser_;
    injectWorkersUser_ = resourceMonitorParamCopy_.injectWorkers_.user_;
    injectWorkersCommand_ = resourceMonitorParamCopy_.injectWorkers_.command_;
    nInjectWorkers_ = resourceMonitorParamCopy_.injectWorkers_.expectedCount_;
    copyWorkersUser_ = resourceMonitorParamCopy_.copyWorkers_.user_;
    copyWorkersCommand_ = resourceMonitorParamCopy_.copyWorkers_.command_;
    nCopyWorkers_ = resourceMonitorParamCopy_.copyWorkers_.expectedCount_;
 
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("sataUser", &sataUser_);
    infoSpace->fireItemAvailable("injectWorkersUser", &injectWorkersUser_);
    infoSpace->fireItemAvailable("injectWorkersCommand", &injectWorkersCommand_);
    infoSpace->fireItemAvailable("nInjectWorkers", &nInjectWorkers_);
    infoSpace->fireItemAvailable("copyWorkersUser", &copyWorkersUser_);
    infoSpace->fireItemAvailable("copyWorkersCommand", &copyWorkersCommand_);
    infoSpace->fireItemAvailable("nCopyWorkers", &nCopyWorkers_);
  }

  void Configuration::
  setupAlarmInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    isProductionSystem_ = alarmParamCopy_.isProductionSystem_;
    careAboutUnwantedEvents_ = alarmParamCopy_.careAboutUnwantedEvents_;
    errorEvents_ = alarmParamCopy_.errorEvents_;
    unwantedEvents_ = alarmParamCopy_.unwantedEvents_;
 
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("isProductionSystem", &isProductionSystem_);
    infoSpace->fireItemAvailable("careAboutUnwantedEvents", &careAboutUnwantedEvents_);
    infoSpace->fireItemAvailable("errorEvents", &errorEvents_);
    infoSpace->fireItemAvailable("unwantedEvents", &unwantedEvents_);
  }

  void Configuration::updateLocalDiskWritingData()
  {
    evf::ParameterSetRetriever smpset(streamConfiguration_);
    diskWriteParamCopy_.streamConfiguration_ = smpset.getAsString();

    diskWriteParamCopy_.fileName_ = fileName_;
    diskWriteParamCopy_.filePath_ = filePath_;
    if ( dbFilePath_.value_.empty() )
      diskWriteParamCopy_.dbFilePath_ = filePath_.value_ + "/log";
    else
      diskWriteParamCopy_.dbFilePath_ = dbFilePath_;
    diskWriteParamCopy_.setupLabel_ = setupLabel_;
    diskWriteParamCopy_.nLogicalDisk_ = nLogicalDisk_;
    diskWriteParamCopy_.maxFileSizeMB_ = maxFileSize_;
    diskWriteParamCopy_.highWaterMark_ = highWaterMark_;
    diskWriteParamCopy_.failHighWaterMark_ = failHighWaterMark_;
    diskWriteParamCopy_.lumiSectionTimeOut_ = utils::secondsToDuration(lumiSectionTimeOut_);
    diskWriteParamCopy_.fileClosingTestInterval_ =
      boost::posix_time::seconds( static_cast<int>(fileClosingTestInterval_) );
    diskWriteParamCopy_.fileSizeTolerance_ = fileSizeTolerance_;
    diskWriteParamCopy_.faultyEventsStream_ = faultyEventsStream_;
    diskWriteParamCopy_.checkAdler32_ = checkAdler32_;

    utils::getStdVector(otherDiskPaths_, diskWriteParamCopy_.otherDiskPaths_);


    streamConfigurationChanged_ = false;
  }

  void Configuration::updateLocalDQMProcessingData()
  {
    dqmParamCopy_.collateDQM_ = collateDQM_;
    dqmParamCopy_.readyTimeDQM_ =
      boost::posix_time::seconds( static_cast<int>(readyTimeDQM_) );
    dqmParamCopy_.useCompressionDQM_ = useCompressionDQM_;
    dqmParamCopy_.compressionLevelDQM_ = compressionLevelDQM_;
    dqmParamCopy_.discardDQMUpdatesForOlderLS_ = discardDQMUpdatesForOlderLS_;
  }

  void Configuration::updateLocalEventServingData()
  {
    eventServeParamCopy_.activeConsumerTimeout_ =
      boost::posix_time::seconds( static_cast<int>(activeConsumerTimeout_) );
    eventServeParamCopy_.consumerQueueSize_ = consumerQueueSize_;
    eventServeParamCopy_.consumerQueuePolicy_ = consumerQueuePolicy_;
    eventServeParamCopy_._DQMactiveConsumerTimeout = 
      boost::posix_time::seconds( static_cast<int>(_DQMactiveConsumerTimeout) );
    eventServeParamCopy_._DQMconsumerQueueSize = _DQMconsumerQueueSize;
    eventServeParamCopy_._DQMconsumerQueuePolicy = _DQMconsumerQueuePolicy;

    // validation
    if (eventServeParamCopy_.consumerQueueSize_ < 1)
    {
      eventServeParamCopy_.consumerQueueSize_ = 1;
    }
    if (eventServeParamCopy_._DQMconsumerQueueSize < 1)
    {
      eventServeParamCopy_._DQMconsumerQueueSize = 1;
    }
  }

  void Configuration::updateLocalQueueConfigurationData()
  {
    queueConfigParamCopy_.commandQueueSize_ = commandQueueSize_;
    queueConfigParamCopy_.dqmEventQueueSize_ = dqmEventQueueSize_;
    queueConfigParamCopy_.dqmEventQueueMemoryLimitMB_ = dqmEventQueueMemoryLimitMB_;
    queueConfigParamCopy_.fragmentQueueSize_ = fragmentQueueSize_;
    queueConfigParamCopy_.fragmentQueueMemoryLimitMB_ = fragmentQueueMemoryLimitMB_;
    queueConfigParamCopy_.registrationQueueSize_ = registrationQueueSize_;
    queueConfigParamCopy_.streamQueueSize_ = streamQueueSize_;
    queueConfigParamCopy_.streamQueueMemoryLimitMB_ = streamQueueMemoryLimitMB_;
    queueConfigParamCopy_.fragmentStoreMemoryLimitMB_ = fragmentStoreMemoryLimitMB_;
  }

  void Configuration::updateLocalWorkerThreadData()
  {
    workerThreadParamCopy_.FPdeqWaitTime_ = utils::secondsToDuration(FPdeqWaitTime_);
    workerThreadParamCopy_.DWdeqWaitTime_ = utils::secondsToDuration(DWdeqWaitTime_);
    workerThreadParamCopy_.DQMEPdeqWaitTime_ = utils::secondsToDuration(DQMEPdeqWaitTime_);

    workerThreadParamCopy_.staleFragmentTimeOut_ = utils::secondsToDuration(staleFragmentTimeOut_);
    workerThreadParamCopy_.monitoringSleepSec_ = utils::secondsToDuration(monitoringSleepSec_);
    workerThreadParamCopy_.throuphputAveragingCycles_ = throuphputAveragingCycles_;
  }

  void Configuration::updateLocalResourceMonitorData()
  {
    resourceMonitorParamCopy_.sataUser_ = sataUser_;
    resourceMonitorParamCopy_.injectWorkers_.user_ = injectWorkersUser_;
    resourceMonitorParamCopy_.injectWorkers_.command_ = injectWorkersCommand_;
    resourceMonitorParamCopy_.injectWorkers_.expectedCount_ = nInjectWorkers_;
    resourceMonitorParamCopy_.copyWorkers_.user_ = copyWorkersUser_;
    resourceMonitorParamCopy_.copyWorkers_.command_ = copyWorkersCommand_;
    resourceMonitorParamCopy_.copyWorkers_.expectedCount_ = nCopyWorkers_;
  }

  void Configuration::updateLocalAlarmData()
  {
    alarmParamCopy_.isProductionSystem_ = isProductionSystem_;
    alarmParamCopy_.careAboutUnwantedEvents_ = careAboutUnwantedEvents_;
    alarmParamCopy_.errorEvents_ = errorEvents_;
    alarmParamCopy_.unwantedEvents_ = unwantedEvents_;
  }

  void Configuration::updateLocalRunNumberData()
  {
    localRunNumber_ = infospaceRunNumber_;
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
