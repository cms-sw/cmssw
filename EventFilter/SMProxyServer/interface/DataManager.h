// $Id: DataManager.h,v 1.1.2.10 2011/03/01 08:32:14 mommsen Exp $
/// @file: DataManager.h 

#ifndef EventFilter_SMProxyServer_DataManager_h
#define EventFilter_SMProxyServer_DataManager_h

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/DataRetrieverMonitorCollection.h"
#include "EventFilter/SMProxyServer/interface/EventRetriever.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/RegistrationQueue.h"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>

#include <map>


namespace smproxy {

  class StateMachine;

  /**
   * Manages the data retrieval
   *
   * $Author: mommsen $
   * $Revision: 1.1.2.10 $
   * $Date: 2011/03/01 08:32:14 $
   */
  
  class DataManager
  {
  public:

    DataManager(StateMachine*);

    ~DataManager();

    /**
     * Start retrieving data
     */
    void start(DataRetrieverParams const&);

    /**
     * Stop retrieving data
     */
    void stop();

    /**
     * Get list of data event consumer queueIDs for given event type.
     * Returns false if the event type is not found.
     */
    bool getQueueIDsFromDataEventRetrievers
    (
      stor::EventConsRegPtr,
      stor::QueueIDs&
    ) const;

    /**
     * Get list of DQM event consumer queueIDs for given event type.
     * Returns false if the event type is not found.
     */
    bool getQueueIDsFromDQMEventRetrievers
    (
      stor::DQMEventConsRegPtr,
      stor::QueueIDs&
    ) const;


  private:

    void activity();
    void doIt();
    bool addEventConsumer(stor::RegPtr);
    bool addDQMEventConsumer(stor::RegPtr);
    void watchDog();
    void checkForStaleConsumers();

    StateMachine* stateMachine_;
    stor::RegistrationQueuePtr registrationQueue_;
    DataRetrieverParams dataRetrieverParams_;

    boost::scoped_ptr<boost::thread> thread_;
    boost::scoped_ptr<boost::thread> watchDogThread_;

    typedef EventRetriever<stor::EventConsumerRegistrationInfo,
                           EventQueueCollectionPtr> DataEventRetriever;
    typedef boost::shared_ptr<DataEventRetriever> DataEventRetrieverPtr;
    typedef std::map<stor::EventConsRegPtr, DataEventRetrieverPtr,
                     stor::utils::ptrComp<stor::EventConsumerRegistrationInfo> > DataEventRetrieverMap;
    DataEventRetrieverMap dataEventRetrievers_;

    typedef EventRetriever<stor::DQMEventConsumerRegistrationInfo,
                           stor::DQMEventQueueCollectionPtr> DQMEventRetriever;
    typedef boost::shared_ptr<DQMEventRetriever> DQMEventRetrieverPtr;
    typedef std::map<stor::DQMEventConsRegPtr, DQMEventRetrieverPtr,
                     stor::utils::ptrComp<stor::DQMEventConsumerRegistrationInfo> > DQMEventRetrieverMap;
    DQMEventRetrieverMap dqmEventRetrievers_;

  };

  typedef boost::shared_ptr<DataManager> DataManagerPtr;
  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_DataManager_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
