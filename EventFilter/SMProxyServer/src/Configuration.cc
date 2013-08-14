// $Id: Configuration.cc,v 1.3 2011/05/09 11:03:34 mommsen Exp $
/// @file: Configuration.cc

#include "EventFilter/SMProxyServer/interface/Configuration.h"

#include <toolbox/net/Utils.h>

#include <sstream>


namespace smproxy
{
  Configuration::Configuration
  (
    xdata::InfoSpace* infoSpace,
    unsigned long instanceNumber
  )
  {
    // default values are used to initialize infospace values,
    // so they should be set first
    setDataRetrieverDefaults(instanceNumber);
    setEventServingDefaults();
    setDQMProcessingDefaults();
    setDQMArchivingDefaults();
    setQueueConfigurationDefaults();
    setAlarmDefaults();

    setupDataRetrieverInfoSpaceParams(infoSpace);
    setupEventServingInfoSpaceParams(infoSpace);
    setupDQMProcessingInfoSpaceParams(infoSpace);
    setupDQMArchivingInfoSpaceParams(infoSpace);
    setupQueueConfigurationInfoSpaceParams(infoSpace);
    setupAlarmInfoSpaceParams(infoSpace);
  }

  struct DataRetrieverParams Configuration::getDataRetrieverParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return dataRetrieverParamCopy_;
  }

  struct stor::EventServingParams Configuration::getEventServingParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return eventServeParamCopy_;
  }

  struct stor::DQMProcessingParams Configuration::getDQMProcessingParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return dqmProcessingParamCopy_;
  }

  struct DQMArchivingParams Configuration::getDQMArchivingParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return dqmArchivingParamCopy_;
  }

  struct QueueConfigurationParams Configuration::getQueueConfigurationParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return queueConfigParamCopy_;
  }
  
  struct AlarmParams Configuration::getAlarmParams() const
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    return alarmParamCopy_;
  }
  
  void Configuration::updateAllParams()
  {
    boost::mutex::scoped_lock sl(generalMutex_);
    updateLocalDataRetrieverData();
    updateLocalEventServingData();
    updateLocalDQMProcessingData();
    updateLocalDQMArchivingData();
    updateLocalQueueConfigurationData();
    updateLocalAlarmData();
  }

  void Configuration::setDataRetrieverDefaults(unsigned long instanceNumber)
  {
    dataRetrieverParamCopy_.smpsInstance_ = instanceNumber;
    dataRetrieverParamCopy_.smRegistrationList_.clear();
    dataRetrieverParamCopy_.allowMissingSM_ = true;
    dataRetrieverParamCopy_.maxConnectionRetries_ = 5;
    dataRetrieverParamCopy_.connectTrySleepTime_ = 10;
    dataRetrieverParamCopy_.headerRetryInterval_ = 5;
    dataRetrieverParamCopy_.retryInterval_ = 1;
    dataRetrieverParamCopy_.sleepTimeIfIdle_ =
      boost::posix_time::milliseconds(100);

    std::string tmpString(toolbox::net::getHostName());
    // strip domainame
    std::string::size_type pos = tmpString.find('.');  
    if (pos != std::string::npos) {  
      std::string basename = tmpString.substr(0,pos);  
      tmpString = basename;
    }
    dataRetrieverParamCopy_.hostName_ = tmpString;
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

  void Configuration::setDQMProcessingDefaults()
  {
    dqmProcessingParamCopy_.collateDQM_ = true;
    dqmProcessingParamCopy_.readyTimeDQM_ = boost::posix_time::seconds(120);
    dqmProcessingParamCopy_.useCompressionDQM_ = true;
    dqmProcessingParamCopy_.compressionLevelDQM_ = 1;
  }

  void Configuration::setDQMArchivingDefaults()
  {
    dqmArchivingParamCopy_.archiveDQM_ = false;
    dqmArchivingParamCopy_.archiveTopLevelFolder_ = "*";
    dqmArchivingParamCopy_.filePrefixDQM_ = "/tmp/DQM";
    dqmArchivingParamCopy_.archiveIntervalDQM_ = 0;
  }

  void Configuration::setQueueConfigurationDefaults()
  {
    queueConfigParamCopy_.registrationQueueSize_ = 128;
    queueConfigParamCopy_.monitoringSleepSec_ = boost::posix_time::seconds(1);
  }

  void Configuration::setAlarmDefaults()
  {
    // set defaults
    alarmParamCopy_.sendAlarms_ = true;
    alarmParamCopy_.corruptedEventRate_ = 0.1;
  }
  
  void Configuration::
  setupDataRetrieverInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults into the xdata variables
    stor::utils::getXdataVector(dataRetrieverParamCopy_.smRegistrationList_, smRegistrationList_);
    allowMissingSM_ = dataRetrieverParamCopy_.allowMissingSM_;
    maxConnectionRetries_ = dataRetrieverParamCopy_.maxConnectionRetries_;
    connectTrySleepTime_ = dataRetrieverParamCopy_.connectTrySleepTime_;
    headerRetryInterval_ = dataRetrieverParamCopy_.headerRetryInterval_;
    retryInterval_ = dataRetrieverParamCopy_.retryInterval_;
    sleepTimeIfIdle_ = dataRetrieverParamCopy_.sleepTimeIfIdle_.total_milliseconds();

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("SMRegistrationList", &smRegistrationList_);
    infoSpace->fireItemAvailable("allowMissingSM", &allowMissingSM_);
    infoSpace->fireItemAvailable("maxConnectionRetries", &maxConnectionRetries_);
    infoSpace->fireItemAvailable("connectTrySleepTime", &connectTrySleepTime_);
    infoSpace->fireItemAvailable("headerRetryInterval", &headerRetryInterval_);
    infoSpace->fireItemAvailable("retryInterval", &retryInterval_);
    infoSpace->fireItemAvailable("sleepTimeIfIdle", &sleepTimeIfIdle_);
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
    infoSpace->fireItemAvailable("activeConsumerTimeout", &activeConsumerTimeout_);
    infoSpace->fireItemAvailable("consumerQueueSize", &consumerQueueSize_);
    infoSpace->fireItemAvailable("consumerQueuePolicy", &consumerQueuePolicy_);
    infoSpace->fireItemAvailable("DQMactiveConsumerTimeout", &_DQMactiveConsumerTimeout);
    infoSpace->fireItemAvailable("DQMconsumerQueueSize", &_DQMconsumerQueueSize);
    infoSpace->fireItemAvailable("DQMconsumerQueuePolicy",&_DQMconsumerQueuePolicy);
  }

  void Configuration::
  setupDQMProcessingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    collateDQM_ = dqmProcessingParamCopy_.collateDQM_;
    readyTimeDQM_ = dqmProcessingParamCopy_.readyTimeDQM_.total_seconds();
    useCompressionDQM_ = dqmProcessingParamCopy_.useCompressionDQM_;
    compressionLevelDQM_ = dqmProcessingParamCopy_.compressionLevelDQM_;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("collateDQM", &collateDQM_);
    infoSpace->fireItemAvailable("readyTimeDQM", &readyTimeDQM_);
    infoSpace->fireItemAvailable("useCompressionDQM", &useCompressionDQM_);
    infoSpace->fireItemAvailable("compressionLevelDQM", &compressionLevelDQM_);
  }

  void Configuration::
  setupDQMArchivingInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    archiveDQM_ = dqmArchivingParamCopy_.archiveDQM_;
    archiveTopLevelFolder_ = dqmArchivingParamCopy_.archiveTopLevelFolder_;
    archiveIntervalDQM_ = dqmArchivingParamCopy_.archiveIntervalDQM_;
    filePrefixDQM_ = dqmArchivingParamCopy_.filePrefixDQM_;

    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("archiveDQM", &archiveDQM_);
    infoSpace->fireItemAvailable("archiveTopLevelFolder", &archiveTopLevelFolder_);
    infoSpace->fireItemAvailable("archiveIntervalDQM", &archiveIntervalDQM_);
    infoSpace->fireItemAvailable("filePrefixDQM", &filePrefixDQM_);
  }
  
  void Configuration::
  setupQueueConfigurationInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    registrationQueueSize_ = queueConfigParamCopy_.registrationQueueSize_;
    monitoringSleepSec_ =
      stor::utils::durationToSeconds(queueConfigParamCopy_.monitoringSleepSec_);
    
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("registrationQueueSize", &registrationQueueSize_);
    infoSpace->fireItemAvailable("monitoringSleepSec", &monitoringSleepSec_);
  }
  
  void Configuration::
  setupAlarmInfoSpaceParams(xdata::InfoSpace* infoSpace)
  {
    // copy the initial defaults to the xdata variables
    sendAlarms_ = alarmParamCopy_.sendAlarms_;
    corruptedEventRate_ = alarmParamCopy_.corruptedEventRate_;
 
    // bind the local xdata variables to the infospace
    infoSpace->fireItemAvailable("sendAlarms", &sendAlarms_);
    infoSpace->fireItemAvailable("corruptedEventRate", &corruptedEventRate_);
  }

  void Configuration::updateLocalDataRetrieverData()
  {
    stor::utils::getStdVector(smRegistrationList_, dataRetrieverParamCopy_.smRegistrationList_);
    dataRetrieverParamCopy_.allowMissingSM_ = allowMissingSM_;
    dataRetrieverParamCopy_.maxConnectionRetries_ = maxConnectionRetries_;
    dataRetrieverParamCopy_.connectTrySleepTime_ = connectTrySleepTime_;
    dataRetrieverParamCopy_.headerRetryInterval_ = headerRetryInterval_;
    dataRetrieverParamCopy_.retryInterval_ = retryInterval_;
    dataRetrieverParamCopy_.sleepTimeIfIdle_ =
      boost::posix_time::milliseconds(sleepTimeIfIdle_);
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

  void Configuration::updateLocalDQMProcessingData()
  {
    dqmProcessingParamCopy_.collateDQM_ = collateDQM_;
    dqmProcessingParamCopy_.readyTimeDQM_ =
      boost::posix_time::seconds( static_cast<int>(readyTimeDQM_) );
    dqmProcessingParamCopy_.useCompressionDQM_ = useCompressionDQM_;
    dqmProcessingParamCopy_.compressionLevelDQM_ = compressionLevelDQM_;
  }

  void Configuration::updateLocalDQMArchivingData()
  {
    dqmArchivingParamCopy_.archiveDQM_ = archiveDQM_;
    dqmArchivingParamCopy_.archiveTopLevelFolder_ = archiveTopLevelFolder_;
    dqmArchivingParamCopy_.archiveIntervalDQM_ = archiveIntervalDQM_;
    dqmArchivingParamCopy_.filePrefixDQM_ = filePrefixDQM_;
  }

  void Configuration::updateLocalQueueConfigurationData()
  {
    queueConfigParamCopy_.registrationQueueSize_ = registrationQueueSize_;
    queueConfigParamCopy_.monitoringSleepSec_ =
      stor::utils::secondsToDuration(monitoringSleepSec_);
  }

  void Configuration::updateLocalAlarmData()
  {
    alarmParamCopy_.sendAlarms_ = sendAlarms_;
    alarmParamCopy_.corruptedEventRate_ = corruptedEventRate_;
  }
  
  void Configuration::actionPerformed(xdata::Event& ispaceEvent)
  {
    boost::mutex::scoped_lock sl(generalMutex_);
  }

} // namespace smproxy

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
