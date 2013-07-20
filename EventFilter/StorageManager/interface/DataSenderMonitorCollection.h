// $Id: DataSenderMonitorCollection.h,v 1.20 2012/04/20 10:48:18 mommsen Exp $
/// @file: DataSenderMonitorCollection.h 

#ifndef EventFilter_StorageManager_DataSenderMonitorCollection_h
#define EventFilter_StorageManager_DataSenderMonitorCollection_h

#include <map>

#include "xdata/UnsignedInteger32.h"
#include "xdata/Integer32.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

namespace stor {

  /**
   * A collection of MonitoredQuantities to track received fragments
   * and events by their source (resource broker, filter unit, etc.)
   *
   * $Author: mommsen $
   * $Revision: 1.20 $
   * $Date: 2012/04/20 10:48:18 $
   */
  
  class DataSenderMonitorCollection : public MonitorCollection
  {
  public:

    /**
     * Key that is used to identify resource brokers.
     */
    struct ResourceBrokerKey
    {
      bool isValid;
      std::string hltURL;
      uint32_t hltTid;
      uint32_t hltInstance;
      uint32_t hltLocalId;
      std::string hltClassName;

      explicit ResourceBrokerKey(I2OChain const& i2oChain)
      {
        if (i2oChain.messageCode() != Header::INVALID)
          {
            isValid = true;
            hltURL = i2oChain.hltURL();
            hltTid = i2oChain.hltTid();
            hltInstance = i2oChain.hltInstance();
            hltLocalId = i2oChain.hltLocalId();
            hltClassName = i2oChain.hltClassName();
          }
        else
          {
            isValid = false;
          }
      }

      bool operator<(ResourceBrokerKey const& other) const
      {
        if (isValid != other.isValid) return isValid < other.isValid;
        if (hltURL != other.hltURL) return hltURL < other.hltURL;
        if (hltTid != other.hltTid) return hltTid < other.hltTid;
        if (hltInstance != other.hltInstance) return hltInstance < other.hltInstance;
        if (hltLocalId != other.hltLocalId) return hltLocalId < other.hltLocalId;
        return hltClassName < other.hltClassName;
      }
    };

    /**
     * Key that is used to identify filter units.
     */
    struct FilterUnitKey
    {
      bool isValid;
      uint32_t fuProcessId;
      uint32_t fuGuid;

      explicit FilterUnitKey(I2OChain const& i2oChain)
      {
        if (i2oChain.messageCode() != Header::INVALID)
          {
            isValid = true;
            fuProcessId = i2oChain.fuProcessId();
            fuGuid = i2oChain.fuGuid();
          }
        else
          {
            isValid = false;
          }
      }

      bool operator<(FilterUnitKey const& other) const
      {
        if (isValid != other.isValid) return isValid < other.isValid;

        // 30-Jun-2009, KAB - we want to keep stats for each filter unit without
        // separating out the output modules.  So, we should only use the process ID
        // in the key.  (The GUID is different for each output module.)
        //if (fuProcessId != other.fuProcessId) return fuProcessId < other.fuProcessId;
        //return fuGuid < other.fuGuid;

        return fuProcessId < other.fuProcessId;
      }
    };

    /**
     * Key that is used to identify output modules.
     */
    typedef uint32_t OutputModuleKey;


    /**
     * The set of information that is kept per output module.
     */
    struct OutputModuleRecord
    {
      std::string name;
      OutputModuleKey id;
      uint32_t initMsgSize;
      MonitoredQuantity fragmentSize;
      MonitoredQuantity eventSize;
      
      OutputModuleRecord(const utils::Duration_t& updateInterval) :
      fragmentSize(updateInterval,boost::posix_time::seconds(10)),
      eventSize(updateInterval,boost::posix_time::seconds(10)) {}
    };
    typedef boost::shared_ptr<OutputModuleRecord> OutModRecordPtr;
    typedef std::map<OutputModuleKey, OutModRecordPtr> OutputModuleRecordMap;

    /**
     * The set of information that is kept per filter unit.
     */
    struct FilterUnitRecord
    {
      FilterUnitKey key;
      OutputModuleRecordMap outputModuleMap;
      MonitoredQuantity shortIntervalEventSize;
      MonitoredQuantity mediumIntervalEventSize;
      MonitoredQuantity dqmEventSize;
      MonitoredQuantity errorEventSize;
      MonitoredQuantity faultyEventSize;
      MonitoredQuantity faultyDQMEventSize;
      MonitoredQuantity dataDiscardCount;
      MonitoredQuantity dqmDiscardCount;
      MonitoredQuantity skippedDiscardCount;
      uint32_t initMsgCount;
      uint32_t lastRunNumber;
      uint64_t lastEventNumber;

      explicit FilterUnitRecord
      (
        FilterUnitKey fuKey,
        const utils::Duration_t& updateInterval
      ) :
        key(fuKey),
        shortIntervalEventSize(updateInterval,boost::posix_time::seconds(10)),
        mediumIntervalEventSize(updateInterval,boost::posix_time::seconds(300)),
        dqmEventSize(updateInterval,boost::posix_time::seconds(10)),
        errorEventSize(updateInterval,boost::posix_time::seconds(10)),
        faultyEventSize(updateInterval,boost::posix_time::seconds(10)),
        faultyDQMEventSize(updateInterval,boost::posix_time::seconds(10)),
        dataDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        dqmDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        skippedDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        initMsgCount(0), lastRunNumber(0), lastEventNumber(0) {}
    };
    typedef boost::shared_ptr<FilterUnitRecord> FURecordPtr;

    /**
     * The set of information that is kept per resource broker.
     */
    struct ResourceBrokerRecord
    {
      ResourceBrokerKey key;
      std::map<FilterUnitKey, FURecordPtr> filterUnitMap;
      OutputModuleRecordMap outputModuleMap;
      MonitoredQuantity eventSize;
      MonitoredQuantity dqmEventSize;
      MonitoredQuantity errorEventSize;
      MonitoredQuantity faultyEventSize;
      MonitoredQuantity faultyDQMEventSize;
      MonitoredQuantity dataDiscardCount;
      MonitoredQuantity dqmDiscardCount;
      MonitoredQuantity skippedDiscardCount;
      uint32_t nExpectedEPs;
      uint32_t initMsgCount;
      uint32_t lastRunNumber;
      uint64_t lastEventNumber;

      explicit ResourceBrokerRecord
      (
        ResourceBrokerKey rbKey,
        const utils::Duration_t& updateInterval
      ) :
        key(rbKey),
        eventSize(updateInterval,boost::posix_time::seconds(10)),
        dqmEventSize(updateInterval,boost::posix_time::seconds(10)),
        errorEventSize(updateInterval,boost::posix_time::seconds(10)),
        faultyEventSize(updateInterval,boost::posix_time::seconds(10)),
        faultyDQMEventSize(updateInterval,boost::posix_time::seconds(10)),
        dataDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        dqmDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        skippedDiscardCount(updateInterval,boost::posix_time::seconds(10)),
        nExpectedEPs(0), initMsgCount(0), lastRunNumber(0), lastEventNumber(0) {}
    };
    typedef boost::shared_ptr<ResourceBrokerRecord> RBRecordPtr;


    /**
     * Results for a given output module.
     */
    struct OutputModuleResult
    {
      std::string name;
      OutputModuleKey id;
      uint32_t initMsgSize;
      MonitoredQuantity::Stats eventStats;
    };
    typedef std::vector< boost::shared_ptr<OutputModuleResult> >
      OutputModuleResultsList;

    /**
     * Results for a given resource broker.
     */
    typedef long long UniqueResourceBrokerID_t;
    struct ResourceBrokerResult
    {
      ResourceBrokerKey key;
      uint32_t filterUnitCount;
      uint32_t initMsgCount;
      uint32_t lastRunNumber;
      uint64_t lastEventNumber;
      MonitoredQuantity::Stats eventStats;
      MonitoredQuantity::Stats dqmEventStats;
      MonitoredQuantity::Stats errorEventStats;
      MonitoredQuantity::Stats faultyEventStats;
      MonitoredQuantity::Stats faultyDQMEventStats;
      MonitoredQuantity::Stats dataDiscardStats;
      MonitoredQuantity::Stats dqmDiscardStats;
      MonitoredQuantity::Stats skippedDiscardStats;
      UniqueResourceBrokerID_t uniqueRBID;
      int outstandingDataDiscardCount;
      int outstandingDQMDiscardCount;

      explicit ResourceBrokerResult(ResourceBrokerKey const& rbKey):
        key(rbKey), filterUnitCount(0), initMsgCount(0),
        lastRunNumber(0), lastEventNumber(0), uniqueRBID(0),
        outstandingDataDiscardCount(0), outstandingDQMDiscardCount(0) {}

      bool operator<(ResourceBrokerResult const& other) const
      {
        return key < other.key;
      }
    };
    typedef boost::shared_ptr<ResourceBrokerResult> RBResultPtr;
    typedef std::vector<RBResultPtr> ResourceBrokerResultsList;

    /**
     * Results for a given filter unit
     */
    struct FilterUnitResult
    {
      FilterUnitKey key;
      uint32_t initMsgCount;
      uint32_t lastRunNumber;
      uint64_t lastEventNumber;
      MonitoredQuantity::Stats shortIntervalEventStats;
      MonitoredQuantity::Stats mediumIntervalEventStats;
      MonitoredQuantity::Stats dqmEventStats;
      MonitoredQuantity::Stats errorEventStats;
      MonitoredQuantity::Stats faultyEventStats;
      MonitoredQuantity::Stats faultyDQMEventStats;
      MonitoredQuantity::Stats dataDiscardStats;
      MonitoredQuantity::Stats dqmDiscardStats;
      MonitoredQuantity::Stats skippedDiscardStats;
      int outstandingDataDiscardCount;
      int outstandingDQMDiscardCount;

      explicit FilterUnitResult(FilterUnitKey const& fuKey):
        key(fuKey), initMsgCount(0), lastRunNumber(0), lastEventNumber(0),
        outstandingDataDiscardCount(0), outstandingDQMDiscardCount(0) {}
    };
    typedef boost::shared_ptr<FilterUnitResult> FUResultPtr;
    typedef std::vector<FUResultPtr> FilterUnitResultsList;


    /**
     * Constructor.
     */
    DataSenderMonitorCollection
    (
      const utils::Duration_t& updateInterval,
      AlarmHandlerPtr
    );

    /**
     * Adds the specified (complete) INIT message to the monitor collection.
     */
    void addInitSample(I2OChain const&);

    /**
     * Adds the specified (complete) Event to the monitor collection.
     */
    void addEventSample(I2OChain const&);

    /**
     * Adds the specified (complete) DQMEvent to the monitor collection.
     */
    void addDQMEventSample(I2OChain const&);

    /**
     * Adds the specified (complete) ErrorEvent to the monitor collection.
     */
    void addErrorEventSample(I2OChain const&);

    /**
     * Adds the specified faulty chain to the monitor collection.
     */
    void addFaultyEventSample(I2OChain const&);

    /**
     * Increments the number of data discard messages tracked by the monitor collection.
     */
    void incrementDataDiscardCount(I2OChain const&);

    /**
     * Increments the number of DQM discard messages tracked by the monitor collection.
     */
    void incrementDQMDiscardCount(I2OChain const&);

    /**
     * Increments the number of skipped discard messages tracked by the monitor collection.
     */
    void incrementSkippedDiscardCount(I2OChain const&);

    /**
     * Fetches the top-level output module statistics.
     */
    OutputModuleResultsList getTopLevelOutputModuleResults() const;

    /**
     * Fetches the resource broker overview statistics.
     */
    ResourceBrokerResultsList getAllResourceBrokerResults() const;

    /**
     * Fetches statistics for a specific resource broker.
     */
    RBResultPtr getOneResourceBrokerResult(UniqueResourceBrokerID_t) const;

    /**
     * Fetches the output module statistics for a specific resource broker.
     */
    OutputModuleResultsList getOutputModuleResultsForRB(UniqueResourceBrokerID_t uniqueRBID) const;

    /**
     * Fetches the filter unit results for a specific resource broker.
     */
    FilterUnitResultsList getFilterUnitResultsForRB(UniqueResourceBrokerID_t uniqueRBID) const;

    /**
     * Return the number of event processors connected.
     */
    size_t getConnectedEPs() const;


  private:

    //Prevent copying of the DataSenderMonitorCollection
    DataSenderMonitorCollection(DataSenderMonitorCollection const&);
    DataSenderMonitorCollection& operator=(DataSenderMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void faultyEventsAlarm(const uint32_t&) const;
    void ignoredDiscardAlarm(const uint32_t&) const;

    bool getAllNeededPointers(I2OChain const& i2oChain,
                              RBRecordPtr& rbRecordPtr,
                              FURecordPtr& fuRecordPtr,
                              OutModRecordPtr& topLevelOutModPtr,
                              OutModRecordPtr& rbSpecificOutModPtr,
                              OutModRecordPtr& fuSpecificOutModPtr);

    bool getRBRecordPointer(I2OChain const& i2oChain,
                            RBRecordPtr& rbRecordPtr);

    bool getFURecordPointer(I2OChain const& i2oChain,
                            RBRecordPtr& rbRecordPtr,
                            FURecordPtr& fuRecordPtr);

    RBRecordPtr getResourceBrokerRecord(ResourceBrokerKey const&);
    UniqueResourceBrokerID_t getUniqueResourceBrokerID(ResourceBrokerKey const&);

    FURecordPtr getFilterUnitRecord(RBRecordPtr&, FilterUnitKey const&);

    OutModRecordPtr getOutputModuleRecord(OutputModuleRecordMap&,
                                          OutputModuleKey const&);

    OutputModuleResultsList buildOutputModuleResults(OutputModuleRecordMap const&) const;

    RBResultPtr buildResourceBrokerResult(RBRecordPtr const&) const;

    void calcStatsForOutputModules(OutputModuleRecordMap& outputModuleMap);

    mutable boost::mutex collectionsMutex_;

    xdata::UnsignedInteger32 connectedRBs_;
    xdata::UnsignedInteger32 connectedEPs_;
    xdata::UnsignedInteger32 activeEPs_;
    xdata::Integer32 outstandingDataDiscards_;
    xdata::Integer32 outstandingDQMDiscards_;
    xdata::UnsignedInteger32 faultyEvents_;
    xdata::UnsignedInteger32 ignoredDiscards_;

    OutputModuleRecordMap outputModuleMap_;

    std::map<ResourceBrokerKey, UniqueResourceBrokerID_t> resourceBrokerIDs_;
    std::map<UniqueResourceBrokerID_t, RBRecordPtr> resourceBrokerMap_;

    const utils::Duration_t updateInterval_;
    AlarmHandlerPtr alarmHandler_;

  };

  bool compareRBResultPtrValues(DataSenderMonitorCollection::RBResultPtr firstValue,
                                DataSenderMonitorCollection::RBResultPtr secondValue);

} // namespace stor

#endif // EventFilter_StorageManager_DataSenderMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
