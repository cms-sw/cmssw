// $Id: DataRetrieverMonitorCollection.h,v 1.3 2011/05/09 11:03:25 mommsen Exp $
/// @file: DataRetrieverMonitorCollection.h 

#ifndef EventFilter_SMProxyServer_DataRetrieverMonitorCollection_h
#define EventFilter_SMProxyServer_DataRetrieverMonitorCollection_h

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/ConnectionID.h"
#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

#include <map>
#include <string>
#include <vector>


namespace smproxy {

  /**
   * A collection of MonitoredQuantities related to data retrieval
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2011/05/09 11:03:25 $
   */
  
  class DataRetrieverMonitorCollection : public stor::MonitorCollection
  {
  public:

    enum ConnectionStatus { CONNECTED, CONNECTION_FAILED, DISCONNECTED, UNKNOWN };

    struct EventStats
    {
      stor::MonitoredQuantity::Stats sizeStats; //kB
      stor::MonitoredQuantity::Stats corruptedEventsStats;
    };
    
    struct SummaryStats
    {
      size_t registeredSMs;
      size_t activeSMs;
      EventStats totals;

      typedef std::pair<stor::RegPtr, EventStats> EventTypeStats;
      typedef std::vector<EventTypeStats> EventTypeStatList;
      EventTypeStatList eventTypeStats;
    };

    typedef std::map<std::string, EventStats> ConnectionStats;

    struct EventTypePerConnectionStats
    {
      stor::RegPtr regPtr;
      ConnectionStatus connectionStatus;
      EventStats eventStats;

      bool operator<(const EventTypePerConnectionStats&) const;
    };
    typedef std::vector<EventTypePerConnectionStats> EventTypePerConnectionStatList;
    
    
    DataRetrieverMonitorCollection
    (
      const stor::utils::Duration_t& updateInterval,
      stor::AlarmHandlerPtr
    );
    
    /**
     * Add a new server connection.
     * Returns an unique connection ID.
     */
    ConnectionID addNewConnection(const stor::RegPtr);

    /**
     * Set status of given connection. Returns false if the ConnectionID is unknown.
     */
    bool setConnectionStatus(const ConnectionID&, const ConnectionStatus&);

    /**
     * Put the event type statistics for the given consumer ID into
     * the passed EventTypePerConnectionStats. Return false if the connection ID is not found.
     */
    bool getEventTypeStatsForConnection(const ConnectionID&, EventTypePerConnectionStats&);

    /**
     * Add a retrieved  sample in Bytes from the given connection.
     * Returns false if the ConnectionID is unknown.
     */
    bool addRetrievedSample(const ConnectionID&, const unsigned int& size);

    /**
     * Increment number of corrupted events received from the given connection.
     * Returns false if the ConnectionID is unknown.
     */
    bool receivedCorruptedEvent(const ConnectionID&);

    /**
     * Write the data retrieval summary statistics into the given struct.
     */
    void getSummaryStats(SummaryStats&) const;

    /**
     * Write the data retrieval statistics for each connection into the given struct.
     */
    void getStatsByConnection(ConnectionStats&) const;

    /**
     * Write the data retrieval statistics for each event type request into the given struct.
     */
    void getStatsByEventTypesPerConnection(EventTypePerConnectionStatList&) const;

    /**
     * Configure the alarm settings
     */
    void configureAlarms(AlarmParams const&);

  private:
    
    struct EventMQ
    {
      stor::MonitoredQuantity size_;       //kB
      stor::MonitoredQuantity corruptedEvents_;

      EventMQ(const stor::utils::Duration_t& updateInterval);
      void getStats(EventStats&) const;
      void calculateStatistics();
      void reset();
    };
    typedef boost::shared_ptr<EventMQ> EventMQPtr;

    struct DataRetrieverMQ
    {
      stor::RegPtr regPtr_;
      ConnectionStatus connectionStatus_;
      EventMQPtr eventMQ_;

      DataRetrieverMQ
      (
        stor::RegPtr,
        const stor::utils::Duration_t& updateInterval
      );
    };

    //Prevent copying of the DataRetrieverMonitorCollection
    DataRetrieverMonitorCollection(DataRetrieverMonitorCollection const&);
    DataRetrieverMonitorCollection& operator=(DataRetrieverMonitorCollection const&);

    const stor::utils::Duration_t updateInterval_;
    stor::AlarmHandlerPtr alarmHandler_;

    AlarmParams alarmParams_;
    EventMQ totals_;

    typedef boost::shared_ptr<DataRetrieverMQ> DataRetrieverMQPtr;
    typedef std::map<ConnectionID, DataRetrieverMQPtr> RetrieverMqMap;
    RetrieverMqMap retrieverMqMap_;

    typedef std::map<std::string, EventMQPtr> ConnectionMqMap;
    ConnectionMqMap connectionMqMap_;

    mutable boost::mutex statsMutex_;
    ConnectionID nextConnectionId_;

    void sendAlarms();
    void checkForCorruptedEvents();
    virtual void do_calculateStatistics();
    virtual void do_reset();
    // virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    // virtual void do_updateInfoSpaceItems();

    class EventTypeMqMap
    {
    public:

      EventTypeMqMap(const stor::utils::Duration_t& updateInterval)
      : updateInterval_(updateInterval) {}

      bool insert(const stor::RegPtr);
      bool addSample(const stor::RegPtr, const double& sizeKB);
      bool receivedCorruptedEvent(const stor::RegPtr);
      void getStats(SummaryStats::EventTypeStatList&) const;
      void calculateStatistics();
      void clear();

    private:

      bool insert(const stor::EventConsRegPtr);
      bool insert(const stor::DQMEventConsRegPtr);
      bool addSample(const stor::EventConsRegPtr, const double& sizeKB);
      bool addSample(const stor::DQMEventConsRegPtr, const double& sizeKB);
      bool receivedCorruptedEvent(const stor::EventConsRegPtr);
      bool receivedCorruptedEvent(const stor::DQMEventConsRegPtr);
      
      typedef std::map<stor::EventConsRegPtr, EventMQPtr,
                       stor::utils::ptrComp<stor::EventConsumerRegistrationInfo>
                       > EventMap;
      EventMap eventMap_;
      
      typedef std::map<stor::DQMEventConsRegPtr, EventMQPtr,
                     stor::utils::ptrComp<stor::DQMEventConsumerRegistrationInfo>
                     > DQMEventMap;
      DQMEventMap dqmEventMap_;
      
      const stor::utils::Duration_t updateInterval_;
    };

    EventTypeMqMap eventTypeMqMap_;

  };

  std::ostream& operator<<(std::ostream&, const DataRetrieverMonitorCollection::ConnectionStatus&);

  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_DataRetrieverMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
