// $Id: DQMEventStore.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DQMEventStore.h 

#ifndef StorageManager_DQMEventStore_h
#define StorageManager_DQMEventStore_h

#include <map>
#include <stack>

#include "boost/shared_ptr.hpp"

#include "IOPool/Streamer/interface/HLTInfo.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventRecord.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"


namespace stor {
  
  /**
   * Stores and collates DQM events
   * Note that this code is not thread safe as it uses a
   * class wide temporary buffer to convert I2OChains
   * into DQMEventMsgViews.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */
  
  class DQMEventStore
  {
  public:
    
    explicit DQMEventStore(DQMEventMonitorCollection&);

    ~DQMEventStore();

    /**
     * Set the DQMProcessingParams to be used.
     * This clears everything in the store.
     */
    void setParameters(DQMProcessingParams const&);

    /**
     * Adds the DQM event found in the I2OChain to
     * the store. If a matching DQMEventRecord is found,
     * the histograms are added unless collateDQM is false.
     */
    void addDQMEvent(I2OChain const&);

    /**
     * Returns true if there is a complete group
     * ready to be served to consumers. In this case
     * DQMEventRecord::GroupRecord holds this record.
     */
    bool getCompletedDQMGroupRecordIfAvailable(DQMEventRecord::GroupRecord&);

    /**
     * Writes and purges all DQMEventRecords hold by the store
     */
    void writeAndPurgeAllDQMInstances();

    /**
     * Clears all DQMEventRecords hold by the DQM store
     */
    void clear();

    /**
     * Checks if the DQM event store is empty
     */
    bool empty()
    { return ( _store.empty() && _recordsReadyToServe.empty() ); }

    
  private:

    //Prevent copying of the DQMEventStore
    DQMEventStore(DQMEventStore const&);
    DQMEventStore& operator=(DQMEventStore const&);

    void addDQMEventToStore(I2OChain const&);

    void addDQMEventToReadyToServe(I2OChain const&);

    void addNextAvailableDQMGroupToReadyToServe(const std::string groupName);

    DQMEventRecordPtr makeDQMEventRecord(I2OChain const&);

    DQMEventMsgView getDQMEventView(I2OChain const&);

    DQMEventRecordPtr getNewestReadyDQMEventRecord(const std::string groupName) const;

    void writeAndPurgeStaleDQMInstances();

    void writeLatestReadyDQMInstance() const;


    DQMProcessingParams _dqmParams;
    DQMEventMonitorCollection& _dqmEventMonColl;

    typedef std::map<DQMKey, DQMEventRecordPtr> DQMEventRecordMap;
    DQMEventRecordMap _store;
    // Always serve the freshest record entry
    std::stack<DQMEventRecord::GroupRecord> _recordsReadyToServe;
    
    std::vector<unsigned char> _tempEventArea;
    
  };
  
} // namespace stor

#endif // StorageManager_DQMEventStore_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
