// $Id: DataSenderMonitorCollection.h,v 1.15 2010/05/17 15:59:09 mommsen Exp $
/// @file: DataSenderMonitorCollection.h 

#ifndef StorageManager_DataSenderMonitorCollection_h
#define StorageManager_DataSenderMonitorCollection_h

#include <map>

#include "xdata/UnsignedInteger32.h"
#include "xdata/Integer32.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

namespace stor {

  class AlarmHandler;


  /**
   * A collection of MonitoredQuantities to track received fragments
   * and events by their source (resource broker, filter unit, etc.)
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2010/05/17 15:59:09 $
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
      unsigned int hltTid;
      unsigned int hltInstance;
      unsigned int hltLocalId;
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
      unsigned int fuProcessId;
      unsigned int fuGuid;

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
      unsigned int initMsgSize;
      //MonitoredQuantity fragmentSize;
      MonitoredQuantity eventSize;
      
      OutputModuleRecord(const utils::duration_t& updateInterval) :
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
      unsigned int initMsgCount;
      unsigned int lastRunNumber;
      unsigned long long lastEventNumber;

      explicit FilterUnitRecord
      (
        FilterUnitKey fuKey,
        const utils::duration_t& updateInterval
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
      unsigned int initMsgCount;
      unsigned int lastRunNumber;
      unsigned long long lastEventNumber;

      explicit ResourceBrokerRecord
      (
        ResourceBrokerKey rbKey,
        const utils::duration_t& updateInterval
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
        initMsgCount(0), lastRunNumber(0), lastEventNumber(0) {}
    };
    typedef boost::shared_ptr<ResourceBrokerRecord> RBRecordPtr;


    /**
     * Results for a given output module.
     */
    struct OutputModuleResult
    {
      std::string name;
      OutputModuleKey id;
      unsigned int initMsgSize;
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
      unsigned int filterUnitCount;
      unsigned int initMsgCount;
      unsigned int lastRunNumber;
      unsigned long long lastEventNumber;
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
      unsigned int initMsgCount;
      unsigned int lastRunNumber;
      unsigned long long lastEventNumber;
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
      const utils::duration_t& updateInterval,
      boost::shared_ptr<AlarmHandler>
    );

    /**
     * Adds the specified fragment to the monitor collection.
     */
    void addFragmentSample(I2OChain const&);

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

  private:

    //Prevent copying of the DataSenderMonitorCollection
    DataSenderMonitorCollection(DataSenderMonitorCollection const&);
    DataSenderMonitorCollection& operator=(DataSenderMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void faultyEventsAlarm(const unsigned int&) const;
    void ignoredDiscardAlarm(const unsigned int&) const;

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

    mutable boost::mutex _collectionsMutex;

    xdata::UnsignedInteger32 _connectedRBs;
    xdata::UnsignedInteger32 _connectedEPs;
    xdata::UnsignedInteger32 _activeEPs;
    xdata::Integer32 _outstandingDataDiscards;
    xdata::Integer32 _outstandingDQMDiscards;
    xdata::UnsignedInteger32 _faultyEvents;
    xdata::UnsignedInteger32 _ignoredDiscards;

    OutputModuleRecordMap _outputModuleMap;

    std::map<ResourceBrokerKey, UniqueResourceBrokerID_t> _resourceBrokerIDs;
    std::map<UniqueResourceBrokerID_t, RBRecordPtr> _resourceBrokerMap;

    const utils::duration_t _updateInterval;
    boost::shared_ptr<AlarmHandler> _alarmHandler;

  };

  bool compareRBResultPtrValues(DataSenderMonitorCollection::RBResultPtr firstValue,
                                DataSenderMonitorCollection::RBResultPtr secondValue);

} // namespace stor

#endif // StorageManager_DataSenderMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
