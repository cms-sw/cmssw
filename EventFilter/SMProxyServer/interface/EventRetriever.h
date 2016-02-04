// $Id: EventRetriever.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: EventRetriever.h 

#ifndef EventFilter_SMProxyServer_EventRetriever_h
#define EventFilter_SMProxyServer_EventRetriever_h

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/ConnectionID.h"
#include "EventFilter/SMProxyServer/interface/DataRetrieverMonitorCollection.h"
#include "EventFilter/SMProxyServer/interface/DQMEventMsg.h"
#include "EventFilter/SMProxyServer/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventStore.h"
#include "EventFilter/StorageManager/interface/EventServerProxy.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>

#include <string>
#include <vector>



namespace smproxy {

  class StateMachine;


  /**
   * Retrieve events from the event server
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:41:54 $
   */

  template<class RegInfo, class QueueCollectionPtr> 
  class EventRetriever
  {
  public:

    typedef boost::shared_ptr<RegInfo> RegInfoPtr;

    EventRetriever
    (
      StateMachine*,
      const RegInfoPtr
    );

    ~EventRetriever();

    /**
     * Add a consumer
     */
    void addConsumer(const RegInfoPtr);

    /**
     * Stop retrieving events
     */
    void stop();

    /**
     * Return the list of QueueIDs attached to the EventRetriever
     */
    const stor::QueueIDs& getQueueIDs() const
    { return queueIDs_; }

    /**
     * Return the number of active connections to SMs
     */
    size_t getConnectedSMCount() const
    { return eventServers_.size(); }

 
  private:

    void activity(const edm::ParameterSet&);
    void doIt(const edm::ParameterSet&);
    void do_stop();
    bool connect(const edm::ParameterSet&);
    void connectToSM(const std::string& sourceURL, const edm::ParameterSet&);
    bool openConnection(const ConnectionID&, const RegInfoPtr);
    bool tryToReconnect();
    void getInitMsg();
    bool getNextEvent(stor::CurlInterface::Content&);
    bool adjustMinEventRequestInterval(const stor::utils::Duration_t&);
    void updateConsumersSetting(const stor::utils::Duration_t&);
    bool anyActiveConsumers(QueueCollectionPtr) const;
    void disconnectFromCurrentSM();
    void processCompletedTopLevelFolders();
    
    //Prevent copying of the EventRetriever
    EventRetriever(EventRetriever const&);
    EventRetriever& operator=(EventRetriever const&);

    StateMachine* stateMachine_;
    const DataRetrieverParams dataRetrieverParams_;
    DataRetrieverMonitorCollection& dataRetrieverMonitorCollection_;

    stor::utils::TimePoint_t nextRequestTime_;
    stor::utils::Duration_t minEventRequestInterval_;

    boost::scoped_ptr<boost::thread> thread_;
    static size_t retrieverCount_;
    size_t instance_;

    typedef stor::EventServerProxy<RegInfo> EventServer;
    typedef boost::shared_ptr<EventServer> EventServerPtr;
    typedef std::map<ConnectionID, EventServerPtr> EventServers;
    EventServers eventServers_;
    typename EventServers::iterator nextSMtoUse_;

    typedef std::vector<ConnectionID> ConnectionIDs;
    ConnectionIDs connectionIDs_;
    mutable boost::mutex connectionIDsLock_;

    stor::utils::TimePoint_t nextReconnectTry_;

    stor::QueueIDs queueIDs_;
    mutable boost::mutex queueIDsLock_;

    stor::DQMEventStore<DQMEventMsg,
                        EventRetriever<RegInfo,QueueCollectionPtr>,
                        StateMachine
                        > dqmEventStore_;
  
  };
  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_EventRetriever_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
