// $Id: DQMEventStore.h,v 1.14 2012/04/20 10:48:18 mommsen Exp $
/// @file: DQMEventStore.h 

#ifndef EventFilter_StorageManager_DQMEventStore_h
#define EventFilter_StorageManager_DQMEventStore_h

#include <map>
#include <queue>

#include <boost/thread/mutex.hpp>

#include "TThread.h"

#include "xcept/Exception.h"
#include "xdaq/ApplicationDescriptor.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMTopLevelFolder.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"


namespace stor {
  
  class DQMEventMonitorCollection;


  /**
   * Stores and collates DQM events
   * Note that this code is not thread safe as it uses a
   * class wide temporary buffer to convert I2OChains
   * into DQMEventMsgViews.
   *
   * $Author: mommsen $
   * $Revision: 1.14 $
   * $Date: 2012/04/20 10:48:18 $
   */

  template<class EventType, class ConnectionType, class StateMachineType>  
  class DQMEventStore
  {
  public:
    
    DQMEventStore
    (
      xdaq::ApplicationDescriptor*,
      DQMEventQueueCollectionPtr,
      DQMEventMonitorCollection&,
      ConnectionType*,
      size_t (ConnectionType::*getExpectedUpdatesCount)() const,
      StateMachineType*,
      void (StateMachineType::*moveToFailedState)(xcept::Exception&),
      AlarmHandlerPtr
    );

    ~DQMEventStore();

    /**
     * Set the DQMProcessingParams to be used.
     * This clears everything in the store.
     */
    void setParameters(DQMProcessingParams const&);

    /**
     * Adds the DQM event found in EventType to the store.
     * If a matching DQMEventRecord is found,
     * the histograms are added unless collateDQM is false.
     */
    void addDQMEvent(EventType const&);

    /**
     * Process completed top level folders, then clear the DQM event store
     */
    void purge();

    /**
     * Clears all data hold by the DQM event store
     */
    void clear();

    /**
     * Checks if the DQM event store is empty
     */
    bool empty()
    { return store_.empty(); }

    void moveToFailedState(xcept::Exception& sentinelException)
    { (stateMachineType_->*moveToFailedState_)(sentinelException); }

    bool doProcessCompletedTopLevelFolders()
    { return processCompletedTopLevelFolders_; }
    
    
  private:

    //Prevent copying of the DQMEventStore
    DQMEventStore(DQMEventStore const&);
    DQMEventStore& operator=(DQMEventStore const&);

    void addDQMEventToStore(EventType const&);
    void addDQMEventToReadyToServe(EventType const&);
    DQMTopLevelFolderPtr makeDQMTopLevelFolder(EventType const&);
    DQMEventMsgView getDQMEventView(EventType const&);
    bool getNextReadyTopLevelFolder(DQMTopLevelFolderPtr&);
    static void processCompletedTopLevelFolders(void* arg);
    bool handleNextCompletedTopLevelFolder();
    void stopProcessingCompletedTopLevelFolders();

    xdaq::ApplicationDescriptor* appDescriptor_;
    DQMProcessingParams dqmParams_;
    DQMEventQueueCollectionPtr dqmEventQueueCollection_;
    DQMEventMonitorCollection& dqmEventMonColl_;
    ConnectionType* connectionType_;
    size_t (ConnectionType::*getExpectedUpdatesCount_)() const;
    StateMachineType* stateMachineType_;
    void (StateMachineType::*moveToFailedState_)(xcept::Exception&);
    AlarmHandlerPtr alarmHandler_;

    typedef std::map<DQMKey, DQMTopLevelFolderPtr> DQMTopLevelFolderMap;
    DQMTopLevelFolderMap store_;
    static boost::mutex storeMutex_;

    TThread* completedFolderThread_;
    bool processCompletedTopLevelFolders_;

    std::vector<unsigned char> tempEventArea_;

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_DQMEventStore_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
