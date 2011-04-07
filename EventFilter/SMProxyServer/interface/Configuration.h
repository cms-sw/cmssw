// $Id: Configuration.h,v 1.1.2.14 2011/03/02 10:59:37 mommsen Exp $
/// @file: Configuration.h 

#ifndef EventFilter_SMProxyServer_Configuration_h
#define EventFilter_SMProxyServer_Configuration_h

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "xdata/InfoSpace.h"
#include "xdata/String.h"
#include "xdata/Integer.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/Vector.h"

#include "boost/thread/mutex.hpp"


namespace smproxy
{
  /**
   * Data structure to hold configuration parameters
   * that are relevant for retrieving events from the SM
   */
  struct DataRetrieverParams
  {
    typedef std::vector<std::string> SMRegistrationList;
    SMRegistrationList smRegistrationList_;
    bool allowMissingSM_;
    uint32_t maxConnectionRetries_;
    uint32_t connectTrySleepTime_;
    uint32_t headerRetryInterval_;
    uint32_t retryInterval_;
    stor::utils::Duration_t sleepTimeIfIdle_;

    // not mapped to infospace params
    uint32_t smpsInstance_;
    std::string hostName_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are relevant for archiving DQM histograms
   */
  struct DQMArchivingParams
  {
    bool archiveDQM_;
    std::string archiveTopLevelFolder_;
    std::string filePrefixDQM_;
    unsigned int archiveIntervalDQM_;
  };

  /**
   * Data structure to hold configuration parameters
   * that are used for the various queues in the system.
   */
  struct QueueConfigurationParams
  {
    uint32_t registrationQueueSize_;
    stor::utils::Duration_t monitoringSleepSec_;
  };

  /**
   * Class for managing configuration information from the infospace
   * and providing local copies of that information that are updated
   * only at requested times.
   *
   * $Author: mommsen $
   * $Revision: 1.1.2.14 $
   * $Date: 2011/03/02 10:59:37 $
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
     * Returns a copy of the event retriever parameters. These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method) or
     * the most recent update of only the event retrieved parameters
     * (see the updateDataRetrieverParams() method).
     */
    struct DataRetrieverParams getDataRetrieverParams() const;

    /**
     * Returns a copy of the DQM processing parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct stor::DQMProcessingParams getDQMProcessingParams() const;

    /**
     * Returns a copy of the DQM archiving parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct DQMArchivingParams getDQMArchivingParams() const;

    /**
     * Returns a copy of the event serving parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct stor::EventServingParams getEventServingParams() const;

    /**
     * Returns a copy of the queue configuration parameters.  These values
     * will be current as of the most recent global update of the local
     * cache from the infospace (see the updateAllParams() method).
     */
    struct QueueConfigurationParams getQueueConfigurationParams() const;

    /**
     * Updates the local copy of all configuration parameters from
     * the infospace.
     */
    void updateAllParams();

    /**
     * Gets invoked when a operation is performed on the infospace
     * that we are interested in knowing about.
     */
    virtual void actionPerformed(xdata::Event& isEvt);


  private:

    void setDataRetrieverDefaults(unsigned long instanceNumber);
    void setEventServingDefaults();
    void setDQMProcessingDefaults();
    void setDQMArchivingDefaults();
    void setQueueConfigurationDefaults();

    void setupDataRetrieverInfoSpaceParams(xdata::InfoSpace*);
    void setupEventServingInfoSpaceParams(xdata::InfoSpace*);
    void setupDQMProcessingInfoSpaceParams(xdata::InfoSpace*);
    void setupDQMArchivingInfoSpaceParams(xdata::InfoSpace*);
    void setupQueueConfigurationInfoSpaceParams(xdata::InfoSpace*);

    void updateLocalDataRetrieverData();
    void updateLocalEventServingData();
    void updateLocalDQMProcessingData();
    void updateLocalDQMArchivingData();
    void updateLocalQueueConfigurationData();

    struct DataRetrieverParams dataRetrieverParamCopy_;
    struct stor::EventServingParams eventServeParamCopy_;
    struct stor::DQMProcessingParams dqmProcessingParamCopy_;
    struct DQMArchivingParams dqmArchivingParamCopy_;
    struct QueueConfigurationParams queueConfigParamCopy_;

    mutable boost::mutex generalMutex_;
    
    xdata::Vector<xdata::String> smRegistrationList_;
    xdata::Boolean allowMissingSM_;
    xdata::UnsignedInteger32 maxConnectionRetries_;
    xdata::UnsignedInteger32 connectTrySleepTime_; // seconds
    xdata::UnsignedInteger32 headerRetryInterval_; // seconds
    xdata::UnsignedInteger32 retryInterval_; // seconds
    xdata::UnsignedInteger32 sleepTimeIfIdle_;  // milliseconds

    xdata::Boolean collateDQM_;
    xdata::Integer readyTimeDQM_;  // seconds
    xdata::Boolean useCompressionDQM_;
    xdata::Integer compressionLevelDQM_;

    xdata::Boolean archiveDQM_;
    xdata::String  archiveTopLevelFolder_;
    xdata::String  filePrefixDQM_;
    xdata::Integer archiveIntervalDQM_;  // lumi sections
    
    xdata::Integer activeConsumerTimeout_;  // seconds
    xdata::Integer consumerQueueSize_;
    xdata::String  consumerQueuePolicy_;
    xdata::Integer _DQMactiveConsumerTimeout;  // seconds
    xdata::Integer _DQMconsumerQueueSize;
    xdata::String  _DQMconsumerQueuePolicy;
    
    xdata::UnsignedInteger32 registrationQueueSize_;
    xdata::Double monitoringSleepSec_;  // seconds

  };

  typedef boost::shared_ptr<Configuration> ConfigurationPtr;

} // namespace smproxy

#endif // EventFilter_SMProxyServer_Configuration_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

