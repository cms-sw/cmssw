// $Id: DataRetrieverMonitorCollection.h,v 1.1.2.8 2011/03/01 08:32:14 mommsen Exp $
/// @file: DataRetrieverMonitorCollection.h 

#ifndef EventFilter_SMProxyServer_DataRetrieverMonitorCollection_h
#define EventFilter_SMProxyServer_DataRetrieverMonitorCollection_h

#include "EventFilter/SMProxyServer/interface/ConnectionID.h"
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
   * $Revision: 1.1.2.8 $
   * $Date: 2011/03/01 08:32:14 $
   */
  
  class DataRetrieverMonitorCollection : public stor::MonitorCollection
  {
  public:

    enum ConnectionStatus { CONNECTED, CONNECTION_FAILED, DISCONNECTED, UNKNOWN };
    
    struct SummaryStats
    {
      size_t registeredSMs;
      size_t activeSMs;
      stor::MonitoredQuantity::Stats sizeStats;         //kB

      typedef std::pair<stor::RegPtr, stor::MonitoredQuantity::Stats> EventTypeStats;
      typedef std::vector<EventTypeStats> EventTypeStatList;
      EventTypeStatList eventTypeStats;
    };

    typedef std::map<std::string, stor::MonitoredQuantity::Stats> ConnectionStats;

    struct EventTypeStats
    {
      stor::RegPtr regPtr;
      ConnectionStatus connectionStatus;
      stor::MonitoredQuantity::Stats sizeStats;         //kB

      bool operator<(const EventTypeStats&) const;
    };
    typedef std::vector<EventTypeStats> EventTypeStatList;
    
    
    explicit DataRetrieverMonitorCollection(const stor::utils::Duration_t& updateInterval);
    
    /**
     * Add a new  server connection.
     * Returns an unique connection ID.
     */
    ConnectionID addNewConnection(const stor::RegPtr);

    /**
     * Set status of given connection. Returns false if the ConnectionID is unknown.
     */
    bool setConnectionStatus(const ConnectionID&, const ConnectionStatus&);

    /**
     * Put the event type statistics for the given consumer ID into
     * the passed EventTypeStats. Return false if the connection ID is not found.
     */
    bool getEventTypeStatsForConnection(const ConnectionID&, EventTypeStats&);

    /**
     * Add a retrieved  sample in Bytes from the given connection.
     * Returns false if the ConnectionID is unknown.
     */
    bool addRetrievedSample(const ConnectionID&, const unsigned int& size);
    
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
    void getStatsByEventTypes(EventTypeStatList&) const;
    

  private:
    
    stor::MonitoredQuantity totalSize_;

    struct DataRetrieverMQ
    {
      stor::RegPtr regPtr_;
      ConnectionStatus connectionStatus_;
      stor::MonitoredQuantity size_;       //kB

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
    typedef boost::shared_ptr<DataRetrieverMQ> DataRetrieverMQPtr;
    typedef std::map<ConnectionID, DataRetrieverMQPtr> RetrieverMqMap;
    RetrieverMqMap retrieverMqMap_;

    typedef std::map<std::string, stor::MonitoredQuantityPtr> ConnectionMqMap;
    ConnectionMqMap connectionMqMap_;

    mutable boost::mutex statsMutex_;
    ConnectionID nextConnectionId_;

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
      void getStats(SummaryStats::EventTypeStatList&) const;
      void calculateStatistics();
      void clear();

    private:

      bool insert(const stor::EventConsRegPtr);
      bool insert(const stor::DQMEventConsRegPtr);
      bool addSample(const stor::EventConsRegPtr, const double& sizeKB);
      bool addSample(const stor::DQMEventConsRegPtr, const double& sizeKB);
      
      typedef std::map<stor::EventConsRegPtr, stor::MonitoredQuantityPtr,
                       stor::utils::ptrComp<stor::EventConsumerRegistrationInfo>
                       > EventMap;
      EventMap eventMap_;
      
      typedef std::map<stor::DQMEventConsRegPtr, stor::MonitoredQuantityPtr,
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
